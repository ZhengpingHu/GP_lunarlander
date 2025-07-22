# yolo_state.py
import numpy as np
import cv2
import time
from ultralytics import YOLO
import torch

class YoloStateEstimator:
    def __init__(self, lander_model_path, terrain_model_path,
                 conf=0.67, tol_x=10, tol_y=10):
        # 加载模型
        self.model_lander = YOLO(lander_model_path, task='obb')
        self.model_terrain = YOLO(terrain_model_path, task='pose')
        self.conf = conf
        # 判定腿尖接触的宽容阈值（像素单位）
        self.tol_x = tol_x
        self.tol_y = tol_y

        # 帧统计
        self.frames_total = 0
        self.frames_lander_ok = 0
        self.frames_terrain_ok = 0

        # 用于速度和角速度估算
        self.prev_x = self.prev_y = self.prev_theta = None
        self.prev_t = None

    def update(self, frame_bgr):
        t = time.time()
        self.frames_total += 1

        # 1. lander OBB 推理
        res_l = self.model_lander.predict(
            source=frame_bgr, imgsz=(608,416), conf=self.conf, verbose=False
        )[0]
        if res_l.obb is None or len(res_l.obb.xywhr) == 0:
            return None
        idx = torch.argmax(res_l.obb.conf)
        x, y, w, h, theta = res_l.obb.xywhr[idx].cpu().numpy()
        self.frames_lander_ok += 1

        # 2. terrain pose 推理
        res_t = self.model_terrain.predict(
            source=frame_bgr, imgsz=(608,416), conf=self.conf, verbose=False
        )[0]
        if res_t.keypoints is None or len(res_t.keypoints.xy) == 0:
            return None
        kpts = res_t.keypoints.xy[0].cpu().numpy()
        self.frames_terrain_ok += 1

        # 3. 速度与角速度估算
        if self.prev_x is None:
            vx = vy = dtheta = 0.0
        else:
            dt = max(t - self.prev_t, 1e-3)
            vx = (x - self.prev_x) / dt
            vy = (y - self.prev_y) / dt
            dtheta = ((theta - self.prev_theta + np.pi) % (2*np.pi) - np.pi) / dt

        # 保存上帧状态
        self.prev_x, self.prev_y, self.prev_theta = x, y, theta
        self.prev_t = t

        # 4. 计算 两腿尖端 坐标 — 简化为左右沿着机身宽度偏移
        dx = (w / 2) * np.sin(theta)
        dy = (w / 2) * np.cos(theta)
        tx1, ty1 = x + dx, y - dy
        tx2, ty2 = x - dx, y + dy

        # 5. 判定腿尖接触
        leg1 = self._check_leg_contact(tx1, ty1, kpts)
        leg2 = self._check_leg_contact(tx2, ty2, kpts)

        # 6. 构造状态向量（8维）
        return np.array([
            x / 608, y / 416, vx / 608, vy / 416,
            theta / np.pi, dtheta / np.pi,
            float(leg1), float(leg2)
        ], dtype=np.float32)

    def _check_leg_contact(self, tx, ty, kpts):
        dx = np.abs(kpts[:,0] - tx)
        idx = np.argmin(dx)
        if dx[idx] < self.tol_x:
            yg = kpts[idx,1]
            if abs(ty - yg) < self.tol_y:
                return True
        return False