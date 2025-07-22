# yolo_state.py
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

    def update(self, frame):
        # 1. lander OBB 推理
        res_lander = self.model_lander(frame, conf=self.conf, track=True)[0]
        if not res_lander or len(res_lander.obb.xy) == 0:
            return None  # 预测失败

        idx = torch.argmax(res_lander.boxes.conf)
        center = res_lander.obb.xy[idx].cpu().numpy()
        theta  = res_lander.obb.theta[idx].cpu().numpy()
        x, y = center

        # 2. 帧差速度估计
        if self.prev_x is None:
            vx = vy = omega = 0.0
        else:
            vx = (x - self.prev_x) / self.interval
            vy = (y - self.prev_y) / self.interval
            omega = (theta - self.prev_theta) / self.interval

        self.prev_x = x
        self.prev_y = y
        self.prev_theta = theta

        # 3. terrain Pose 推理
        res_terrain = self.model_terrain(frame, conf=self.conf, track=True)[0]
        if not res_terrain or res_terrain.keypoints is None:
            return None

        keypoints = res_terrain.keypoints.xy[0].cpu().numpy()  # (K,2)
        keypoints = sorted(keypoints, key=lambda p: p[0])      # 按x排序

        # 4. 支架端点计算（简化为 ±dx, -dy 旋转）
        dx, dy = 20, 30  # 支架相对中心偏移（像素），你可按需要调整
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        def rotate(px, py):
            return x + cos_t * px - sin_t * py, y + sin_t * px + cos_t * py
        left_tip  = rotate(-dx, -dy)
        right_tip = rotate(+dx, -dy)

        def is_touching(pt):
            for i in range(len(keypoints)-1):
                x1, y1 = keypoints[i]
                x2, y2 = keypoints[i+1]
                dist = point_to_segment_dist(pt, (x1,y1), (x2,y2))
                if dist < 5:  # 阈值（像素）
                    return True
            return False

        leg1 = 1 if is_touching(left_tip)  else 0
        leg2 = 1 if is_touching(right_tip) else 0

        return np.array([x, y, vx, vy, theta, omega, leg1, leg2], dtype=np.float32)

# 计算点到线段的距离
def point_to_segment_dist(p, a, b):
    pa, ba = np.array(p)-a, b-a
    h = np.clip(np.dot(pa, ba) / (np.dot(ba, ba) + 1e-6), 0.0, 1.0)
    return np.linalg.norm(pa - h*ba)
