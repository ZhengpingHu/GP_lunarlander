# yolo_state.py
import os
os.environ['ULTRALYTICS_VERBOSE'] = 'False'  # 全局关闭 YOLO 控制台输出

import torch
import numpy as np
from ultralytics import YOLO

class YoloStateEstimator:
    def __init__(self, lander_model_path, terrain_model_path, conf=0.67, fps=60.0):
        self.model_lander = YOLO(lander_model_path, task='obb')
        self.model_terrain = YOLO(terrain_model_path, task='pose')
        self.conf = conf
        self.interval = 1.0 / fps

        self.prev_x = None
        self.prev_y = None
        self.prev_theta = None

        # 用于帧统计
        self.frames_total = 0
        self.frames_lander_ok = 0
        self.frames_terrain_ok = 0

    def update(self, frame):
        self.frames_total += 1

        # === 1. Lander OBB 推理 ===
        res_lander = self.model_lander.predict(frame, conf=self.conf, verbose=False)[0]

        if (
            res_lander is None or
            res_lander.boxes is None or
            res_lander.obb is None or
            res_lander.obb.xywhr is None or
            len(res_lander.obb.xywhr) == 0
        ):
            return None  # 识别失败，跳过

        self.frames_lander_ok += 1

        idx = torch.argmax(res_lander.boxes.conf)
        xywhr = res_lander.obb.xywhr[idx].cpu().numpy()
        x, y, w, h, theta = xywhr

        # === 2. 速度估计（帧差） ===
        if self.prev_x is None:
            vx = vy = omega = 0.0
        else:
            vx = (x - self.prev_x) / self.interval
            vy = (y - self.prev_y) / self.interval
            omega = (theta - self.prev_theta) / self.interval

        self.prev_x = x
        self.prev_y = y
        self.prev_theta = theta

        # === 3. 地形 Pose 推理 ===
        res_terrain = self.model_terrain.predict(frame, conf=self.conf, verbose=False)[0]

        terrain_ok = (
            res_terrain is not None and
            res_terrain.keypoints is not None and
            res_terrain.keypoints.xy is not None and
            len(res_terrain.keypoints.xy[0]) > 0
        )
        if terrain_ok:
            self.frames_terrain_ok += 1
        else:
            return None  # 地形关键点识别失败

        keypoints = res_terrain.keypoints.xy[0].cpu().numpy()
        keypoints = sorted(keypoints, key=lambda p: p[0])  # 按 x 排序

        # === 4. 支架端点计算（绕中心旋转）===
        dx, dy = 20, 30  # 可根据实际 lander 尺寸修改
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        def rotate(px, py):
            return x + cos_t * px - sin_t * py, y + sin_t * px + cos_t * py

        left_tip = rotate(-dx, -dy)
        right_tip = rotate(+dx, -dy)

        # === 5. 接地判断 ===
        def is_touching(pt):
            for i in range(len(keypoints) - 1):
                x1, y1 = keypoints[i]
                x2, y2 = keypoints[i + 1]
                dist = point_to_segment_dist(pt, (x1, y1), (x2, y2))
                if dist < 5:  # 5 像素以内认为接地
                    return True
            return False

        leg1 = 1 if is_touching(left_tip) else 0
        leg2 = 1 if is_touching(right_tip) else 0

        return np.array([x, y, vx, vy, theta, omega, leg1, leg2], dtype=np.float32)

# === 辅助函数：点到线段距离 ===
def point_to_segment_dist(p, a, b):
    pa = np.array(p) - a
    ba = b - a
    h = np.clip(np.dot(pa, ba) / (np.dot(ba, ba) + 1e-6), 0.0, 1.0)
    return np.linalg.norm(pa - h * ba)
