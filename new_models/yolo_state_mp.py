# -*- coding: utf-8 -*-
import numpy as np
import cv2
import time
from ultralytics import YOLO
import torch

class YoloStateEstimator:
    def __init__(self, lander_model_path, terrain_model_path,
                 conf=0.67, tol_x=10, tol_y=10, device=None):
        """
        device: 传给YOLO的device字符串，如 '0', 'cpu'。None则用默认。
        """
        self.model_lander  = YOLO(lander_model_path, task='obb')
        self.model_terrain = YOLO(terrain_model_path, task='pose')
        if device is not None:
            # ultralytics>=8.2: model.to(device) / predict(device=...)
            try:
                self.model_lander.to(device)
                self.model_terrain.to(device)
            except Exception:
                pass

        self.conf  = conf
        self.tol_x = tol_x
        self.tol_y = tol_y

        # 计数器 & 历史
        self._init_counters()
        self.reset_history()

    # ---------- 计数器相关 ----------
    def _init_counters(self):
        self.frames_total      = 0
        self.frames_lander_ok  = 0
        self.frames_terrain_ok = 0

    def reset_counters(self):
        self._init_counters()

    # ---------- 历史（速度估计） ----------
    def reset_history(self):
        self.prev_x = self.prev_y = self.prev_theta = None
        self.prev_t = None

    # ---------- 主更新函数 ----------
    def update(self, frame_bgr):
        """
        输入: BGR图像 (H,W,3)
        输出: 8维状态向量或 None
        """
        t_now = time.time()
        self.frames_total += 1

        # 1. lander OBB 推理
        res_l = self.model_lander.predict(
            source=frame_bgr,
            imgsz=(608, 416),
            conf=self.conf,
            verbose=False
        )[0]

        if res_l.obb is None or len(res_l.obb.xywhr) == 0:
            return None

        idx = torch.argmax(res_l.obb.conf)
        x, y, w, h, theta = res_l.obb.xywhr[idx].cpu().numpy().astype(float)
        self.frames_lander_ok += 1

        # 2. terrain pose 推理
        res_t = self.model_terrain.predict(
            source=frame_bgr,
            imgsz=(608, 416),
            conf=self.conf,
            verbose=False
        )[0]

        if res_t.keypoints is None or len(res_t.keypoints.xy) == 0:
            return None

        kpts = res_t.keypoints.xy[0].cpu().numpy().astype(float)  # Nx2
        self.frames_terrain_ok += 1

        # 3. 速度/角速度
        if self.prev_x is None:
            vx = vy = dtheta = 0.0
        else:
            dt = max(t_now - self.prev_t, 1e-3)
            vx = (x - self.prev_x) / dt
            vy = (y - self.prev_y) / dt
            # wrap to [-pi, pi]
            dtheta = ((theta - self.prev_theta + np.pi) % (2 * np.pi) - np.pi) / dt

        self.prev_x, self.prev_y, self.prev_theta = x, y, theta
        self.prev_t = t_now

        # 4. 两腿尖端位置（简单近似）
        dx = (w / 2.0) * np.sin(theta)
        dy = (w / 2.0) * np.cos(theta)
        tx1, ty1 = x + dx, y - dy
        tx2, ty2 = x - dx, y + dy

        # 5. 判定接地
        leg1 = self._check_leg_contact(tx1, ty1, kpts)
        leg2 = self._check_leg_contact(tx2, ty2, kpts)

        # 6. 构造 8 维状态
        state = np.array([
            x / 608.0,           # 0
            y / 416.0,           # 1
            vx / 608.0,          # 2
            vy / 416.0,          # 3
            theta / np.pi,       # 4
            dtheta / np.pi,      # 5
            float(leg1),         # 6
            float(leg2)          # 7
        ], dtype=np.float32)

        return state

    # ---------- 工具 ----------
    def _check_leg_contact(self, tx, ty, kpts):
        """
        简单用x最接近的地形点，判断y差值是否在容差内。
        """
        dx = np.abs(kpts[:, 0] - tx)
        idx = np.argmin(dx)
        if dx[idx] < self.tol_x:
            yg = kpts[idx, 1]
            if abs(ty - yg) < self.tol_y:
                return True
        return False
