# -*- coding: utf-8 -*-
import numpy as np
import cv2
import time
from ultralytics import YOLO
import torch
import warnings

# ================= 輔助工具類：卡爾曼濾波器 =================
# 簡單的 1D 卡爾曼濾波器，用於平滑單個變量
class KalmanFilter:
    def __init__(self, process_var, measurement_var):
        self.state_est = 0.0
        self.state_err = 1.0
        self.process_var = process_var
        self.measurement_var = measurement_var

    def update(self, measurement):
        # 預測
        pred_state = self.state_est
        pred_err = self.state_err + self.process_var
        
        # 更新
        kalman_gain = pred_err / (pred_err + self.measurement_var)
        self.state_est = pred_state + kalman_gain * (measurement - pred_state)
        self.state_err = (1 - kalman_gain) * pred_err
        return self.state_est

# =======================================================

class YoloStateEstimator:
    def __init__(self, lander_model_path, terrain_model_path,
                 conf=0.67, tol_x=10, tol_y=10, device=None):
        self.model_lander = YOLO(lander_model_path, task='obb')
        self.model_terrain = YOLO(terrain_model_path, task='pose')
        if device is not None:
            try:
                self.model_lander.to(device)
                self.model_terrain.to(device)
            except Exception:
                warnings.warn(f"Failed to move YOLO models to device {device}, falling back to default.", UserWarning)

        self.conf = conf
        self.tol_x = tol_x
        self.tol_y = tol_y

        # 卡爾曼濾波器實例，用於平滑 x, y, theta
        self.kf_x = KalmanFilter(process_var=1e-5, measurement_var=0.5)
        self.kf_y = KalmanFilter(process_var=1e-5, measurement_var=0.5)
        self.kf_theta = KalmanFilter(process_var=1e-5, measurement_var=0.5)

        # 狀態歷史（速度估計）
        self.prev_x = self.prev_y = self.prev_theta = None
        self.prev_t = None
        
        # 计数器
        self._init_counters()

    # ---------- 计数器相关 ----------
    def _init_counters(self):
        self.frames_total = 0
        self.frames_lander_ok = 0
        self.frames_terrain_ok = 0

    def reset_counters(self):
        self._init_counters()

    # ---------- 历史（速度估计） ----------
    def reset_history(self):
        self.prev_x = self.prev_y = self.prev_theta = None
        self.prev_t = None
        # 重置卡爾曼濾波器
        self.kf_x = KalmanFilter(process_var=1e-5, measurement_var=0.5)
        self.kf_y = KalmanFilter(process_var=1e-5, measurement_var=0.5)
        self.kf_theta = KalmanFilter(process_var=1e-5, measurement_var=0.5)

    # ---------- 主更新函数 ----------
    def update(self, frame_bgr):
        t_now = time.time()
        self.frames_total += 1
        
        res_l = self.model_lander.predict(
            source=frame_bgr, imgsz=(608, 416), conf=self.conf, verbose=False
        )[0]
        if res_l.obb is None or len(res_l.obb.xywhr) == 0:
            return None
        idx = torch.argmax(res_l.obb.conf)
        x_raw, y_raw, w, h, theta_raw = res_l.obb.xywhr[idx].cpu().numpy().astype(float)
        self.frames_lander_ok += 1

        res_t = self.model_terrain.predict(
            source=frame_bgr, imgsz=(608, 416), conf=self.conf, verbose=False
        )[0]
        if res_t.keypoints is None or len(res_t.keypoints.xy) == 0:
            return None
        kpts = res_t.keypoints.xy[0].cpu().numpy().astype(float)
        self.frames_terrain_ok += 1

        # 狀態平滑：使用卡爾曼濾波器
        x = self.kf_x.update(x_raw)
        y = self.kf_y.update(y_raw)
        theta = self.kf_theta.update(theta_raw)
        
        vx = vy = dtheta = 0.0
        if self.prev_x is not None:
            dt = max(t_now - self.prev_t, 1/60.0)
            vx = (x - self.prev_x) / dt
            vy = (y - self.prev_y) / dt
            angle_diff = (theta - self.prev_theta + np.pi) % (2 * np.pi) - np.pi
            dtheta = angle_diff / dt
        
        self.prev_x, self.prev_y, self.prev_theta = x, y, theta
        self.prev_t = t_now

        dx = (w / 2.0) * np.sin(theta)
        dy = (w / 2.0) * np.cos(theta)
        tx1, ty1 = x + dx, y - dy
        tx2, ty2 = x - dx, y + dy

        # 改进版：使用线性插值来判断着陆。
        leg1 = self._check_leg_contact_interp(tx1, ty1, kpts)
        leg2 = self._check_leg_contact_interp(tx2, ty2, kpts)

        state = np.array([
            x / 608.0,
            y / 416.0,
            vx / 608.0,
            vy / 416.0,
            theta / np.pi,
            dtheta / np.pi,
            float(leg1),
            float(leg2)
        ], dtype=np.float32)

        return state

    # ---------- 工具 (改进的着陆判断) ----------
    def _check_leg_contact_interp(self, tx, ty, kpts):
        """
        使用线性插值来判断着陆。
        """
        sorted_kpts = kpts[np.argsort(kpts[:, 0])]
        
        if tx < sorted_kpts[0, 0] or tx > sorted_kpts[-1, 0]:
            return False

        idx = np.searchsorted(sorted_kpts[:, 0], tx)
        p1 = sorted_kpts[idx-1]
        p2 = sorted_kpts[idx]
        
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]

        if x2 - x1 == 0:
            yg_interp = y1
        else:
            yg_interp = y1 + (y2 - y1) * (tx - x1) / (x2 - x1)
        
        if abs(ty - yg_interp) < self.tol_y:
            return True
        return False