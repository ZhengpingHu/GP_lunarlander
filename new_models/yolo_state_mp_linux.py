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
                 conf=0.67, tol_x=10, tol_y=10, device=None, smooth_alpha=0.5):
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
        self.smooth_alpha = smooth_alpha

        # 狀態歷史（卡爾曼濾波器或簡單的平滑）
        self.reset_history()
        # 為了簡化，這裡使用指數移動平均(EWMA)代替卡爾曼濾波器
        # EWMA 可以達到很好的平滑效果，且實現更簡單
        self.smoothed_x = None
        self.smoothed_y = None
        self.smoothed_theta = None

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
        self.smoothed_x = None
        self.smoothed_y = None
        self.smoothed_theta = None

    # ---------- 主更新函数 ----------
    def update(self, frame_bgr):
        """
        输入: BGR图像 (H,W,3)
        输出: 8维状态向量或 None
        """
        self.frames_total += 1
        t_now = time.time()
        
        # 1. lander OBB 推理
        res_l = self.model_lander.predict(
            source=frame_bgr, imgsz=(608, 416), conf=self.conf, verbose=False
        )[0]
        if res_l.obb is None or len(res_l.obb.xywhr) == 0:
            return None
        idx = torch.argmax(res_l.obb.conf)
        x_raw, y_raw, w, h, theta_raw = res_l.obb.xywhr[idx].cpu().numpy().astype(float)
        self.frames_lander_ok += 1

        # 2. terrain pose 推理
        res_t = self.model_terrain.predict(
            source=frame_bgr, imgsz=(608, 416), conf=self.conf, verbose=False
        )[0]
        if res_t.keypoints is None or len(res_t.keypoints.xy) == 0:
            return None
        kpts = res_t.keypoints.xy[0].cpu().numpy().astype(float)
        self.frames_terrain_ok += 1

        # 3. 状态平滑
        if self.smoothed_x is None:
            self.smoothed_x = x_raw
            self.smoothed_y = y_raw
            self.smoothed_theta = theta_raw
        else:
            self.smoothed_x = self.smooth_alpha * x_raw + (1 - self.smooth_alpha) * self.smoothed_x
            self.smoothed_y = self.smooth_alpha * y_raw + (1 - self.smooth_alpha) * self.smoothed_y
            self.smoothed_theta = self.smooth_alpha * theta_raw + (1 - self.smooth_alpha) * self.smoothed_theta
        
        # 使用平滑后的值
        x, y, theta = self.smoothed_x, self.smoothed_y, self.smoothed_theta

        # 4. 速度/角速度
        vx = vy = dtheta = 0.0
        if self.prev_x is not None:
            # 限制最小时间步长以避免速度過大
            dt = max(t_now - self.prev_t, 1/60.0) # 假設最小幀率為60fps
            vx = (x - self.prev_x) / dt
            vy = (y - self.prev_y) / dt
            # 角度差值處理，確保在 [-pi, pi] 範圍內
            angle_diff = (theta - self.prev_theta + np.pi) % (2 * np.pi) - np.pi
            dtheta = angle_diff / dt
        
        self.prev_x, self.prev_y, self.prev_theta = x, y, theta
        self.prev_t = t_now

        # 5. 两腿尖端位置（简单近似）
        dx = (w / 2.0) * np.sin(theta)
        dy = (w / 2.0) * np.cos(theta)
        tx1, ty1 = x + dx, y - dy
        tx2, ty2 = x - dx, y + dy

        # 6. 判定接地 (改进版：使用线性插值)
        leg1 = self._check_leg_contact_interp(tx1, ty1, kpts)
        leg2 = self._check_leg_contact_interp(tx2, ty2, kpts)

        # 7. 构造 8 维状态 (歸一化與原始碼一致)
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
        # 按 x 坐标排序关键点，以便进行插值
        sorted_kpts = kpts[np.argsort(kpts[:, 0])]
        
        if tx < sorted_kpts[0, 0] or tx > sorted_kpts[-1, 0]:
            return False

        # 找到 tx 所在的两个关键点
        idx = np.searchsorted(sorted_kpts[:, 0], tx)
        p1 = sorted_kpts[idx-1]
        p2 = sorted_kpts[idx]
        
        # 线性插值计算地形的 y 值
        # y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]

        if x2 - x1 == 0:
            yg_interp = y1
        else:
            yg_interp = y1 + (y2 - y1) * (tx - x1) / (x2 - x1)
        
        # 判断腿部 y 坐标是否在容差内
        if abs(ty - yg_interp) < self.tol_y:
            return True
        return False