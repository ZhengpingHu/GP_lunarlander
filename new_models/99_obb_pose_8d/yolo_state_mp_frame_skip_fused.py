#!/usr/bin/env python3
# yolo_state_mp_frame_skip_fused.py
# -*- coding: utf-8 -*-
"""
两阶段状态估计器（OBB + Pose + 地形接地）+ 角度变化双约束（软+硬）
- OBB 用于定位+裁剪
- Pose 用于角度（脚尖连线 + PCA兜底）
- 地形高度场（Canny+插值+平滑） + 脚尖间隙 + 迟滞/去抖 → 左右脚接地
- 提供 frame-skip 外推：predict_only()
- 输出 8 维状态： [cosθ, sinθ, x_norm, y_norm, vx_norm, vy_norm, legL, legR]
- 新增：角度变化的 “软上限” + “硬上限”：
  * 软上限（动态）：随像素速度 v_pix 变化的角速度上限：ω_soft = soft_floor + soft_gain * v_pix
  * 硬上限（常数）：max_spin_hard_deg_s，超过直接认为异常并回退（reject）
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import math, time
import numpy as np
import cv2
from ultralytics import YOLO

# ----------------- 角度工具 -----------------
def wrap_pi(a: float) -> float:
    while a <= -math.pi: a += 2*math.pi
    while a >  math.pi: a -= 2*math.pi
    return a

def ang_diff(a: float, b: float) -> float:
    return wrap_pi(a - b)

def ang_axis_diff(a: float, b: float) -> float:
    d = abs(ang_diff(a, b))
    return min(d, abs(wrap_pi(d - math.pi)))

def near_vertical(theta, thresh_deg=20.0):
    d = min(abs(wrap_pi(theta - math.pi/2)), abs(wrap_pi(theta + math.pi/2)))
    return d < math.radians(thresh_deg)

# 鲁棒测角：脚尖 + PCA 兜底
def robust_angle_from_kpts(k_org, use_pca_fallback=True, min_dx_px=3.0):
    """
    k_org: (K,2)，至少前4点分别为 [*, *, left_foot, right_foot]
    返回：theta (rad) 或 None
    """
    if k_org is None or k_org.shape[0] < 4 or not np.all(np.isfinite(k_org[:4])):
        return None

    feet = k_org[2:4].astype(np.float32).copy()
    if feet.shape[0] < 2 or not np.all(np.isfinite(feet)):
        return None

    order = np.argsort(feet[:, 0])
    if order.size < 2:
        return None
    left_foot, right_foot = feet[order[0]], feet[order[1]]

    dx = float(right_foot[0] - left_foot[0])
    dy = float(right_foot[1] - left_foot[1])
    base_dx = abs(dx)

    if base_dx >= min_dx_px:
        return wrap_pi(math.atan2(dy, dx))

    if not use_pca_fallback:
        return wrap_pi(math.atan2(dy, dx))

    P = k_org[:4, :2].astype(np.float32)
    if not np.all(np.isfinite(P)):
        return wrap_pi(math.atan2(dy, dx))

    mu = P.mean(axis=0, keepdims=True)
    X = P - mu
    C = (X.T @ X) / max(len(P) - 1, 1)
    eigvals, eigvecs = np.linalg.eig(C)
    v = eigvecs[:, int(np.argmax(eigvals))]

    leftmost  = P[np.argmin(P[:, 0])]
    rightmost = P[np.argmax(P[:, 0])]
    ref = (rightmost - leftmost).astype(np.float32)
    if np.dot(v, ref) < 0:
        v = -v

    theta = math.atan2(float(v[1]), float(v[0]))
    return wrap_pi(theta)

# ----------------- 地形高度场提取 -----------------
class TerrainTracker:
    def __init__(self, smooth_kernel=9, update_period=3, canny1=50, canny2=120):
        self.smooth_kernel = smooth_kernel
        self.update_period = max(1, int(update_period))
        self.canny1 = canny1
        self.canny2 = canny2
        self.last_height = None   # np.ndarray shape (W,)
        self.frame_ctr = 0

    def _extract_heightfield(self, frame_rgb: np.ndarray) -> np.ndarray:
        H, W = frame_rgb.shape[:2]
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(gray, self.canny1, self.canny2, L2gradient=True)

        height = np.full(W, np.nan, dtype=np.float32)
        ys, xs = np.where(edges > 0)
        if len(xs):
            max_y_per_x = {}
            for x, y in zip(xs, ys):
                if x not in max_y_per_x or y > max_y_per_x[x]:
                    max_y_per_x[x] = y
            for x, y in max_y_per_x.items():
                height[x] = float(y)

        # 缺失填充
        if np.isnan(height).any():
            idx = np.arange(W)
            valid = ~np.isnan(height)
            if valid.any():
                height = np.interp(idx, idx[valid], height[valid])
            else:
                height[:] = H-1

        # 1D 高斯平滑两遍（float32 兼容）
        k = self.smooth_kernel | 1  # 保证奇数
        height_img = height.reshape(1, -1)
        height_img = cv2.GaussianBlur(height_img, (k, 1), 0, borderType=cv2.BORDER_REPLICATE)
        height_img = cv2.GaussianBlur(height_img, (k, 1), 0, borderType=cv2.BORDER_REPLICATE)
        height = height_img.reshape(-1)
        return height

    def update(self, frame_rgb: np.ndarray) -> np.ndarray:
        self.frame_ctr += 1
        if (self.last_height is None) or (self.frame_ctr % self.update_period == 0):
            self.last_height = self._extract_heightfield(frame_rgb)
        return self.last_height

    def query(self, x: float) -> float:
        if self.last_height is None:
            return np.nan
        W = len(self.last_height)
        if x <= 0: return float(self.last_height[0])
        if x >= W-1: return float(self.last_height[-1])
        x0 = int(np.floor(x)); x1 = x0 + 1
        t = x - x0
        return float((1-t)*self.last_height[x0] + t*self.last_height[x1])

# ----------------- 坐标/裁剪工具 -----------------
def affine_and_crop_bgr(frame_rgb, cx, cy, w, h, theta_rad, pad_ratio):
    H, W = frame_rgb.shape[:2]
    angle_deg = -theta_rad * 180.0 / math.pi
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rotated = cv2.warpAffine(frame_rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    w_pad = w * (1.0 + 2.0 * pad_ratio)
    h_pad = h * (1.0 + 2.0 * pad_ratio)
    x0 = int(round(cx - w_pad/2)); y0 = int(round(cy - h_pad/2))
    x1 = int(round(cx + w_pad/2)); y1 = int(round(cy + h_pad/2))
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x1), min(H, y1)
    if x1c <= x0c or y1c <= y0c:
        return None, None
    crop_rgb = rotated[y0c:y1c, x0c:x1c].copy()
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR).copy()
    return crop_bgr, (M, x0c, y0c)

# ----------------- 主类：FusedStateEstimator -----------------
class FusedStateEstimator:
    """
    两阶段状态估计器（适配 frame-skip）
    - update_full(frame_bgr): 运行 OBB+Pose，更新并返回 8 维状态
    - predict_only(): 不做 YOLO，基于上次速度做短期外推
    - 状态向量: [cosθ, sinθ, x_norm, y_norm, vx_norm, vy_norm, legL, legR]
    - 新增：角度变化软/硬上限 + 平滑
    """
    def __init__(self,
                 obb_model_path: str,
                 pose_model_path: str,
                 conf_obb: float = 0.25,
                 conf_pose: float = 0.20,
                 imgsz_obb: int = 640,
                 imgsz_pose: int = 384,
                 base_pad: float = 0.25,
                 gate_deg: float = 120.0,
                 smooth_alpha: float = 0.2,
                 # 角度变化约束
                 max_spin_hard_deg_s: float = 220.0,  # 硬上限（单侧满推理论最大旋转速率）
                 soft_floor_deg_s: float = 30.0,      # 软上限下限（几乎静止时允许的角速）
                 soft_gain_deg_per_pix_s: float = 0.50, # 软上限随像素速度增长的斜率
                 device: str = "auto",
                 frameskip_period: int = 5):
        # 设备只用于 torch 同步/控制，YOLO 内部自行选择（PT）
        if device == "auto":
            self.device = "cuda" if cv2.ocl.haveOpenCL() else "cpu"
        else:
            self.device = device

        self.obb = YOLO(obb_model_path)
        self.pose = YOLO(pose_model_path)

        self.conf_obb = conf_obb
        self.conf_pose = conf_pose
        self.imgsz_obb = imgsz_obb
        self.imgsz_pose = imgsz_pose
        self.base_pad = base_pad
        self.gate_rad = math.radians(gate_deg)
        self.alpha = float(np.clip(smooth_alpha, 0.0, 1.0))

        # 角速度约束参数
        self.max_spin_hard_deg_s = float(max_spin_hard_deg_s)
        self.soft_floor_deg_s = float(soft_floor_deg_s)
        self.soft_gain_deg_per_pix_s = float(soft_gain_deg_per_pix_s)

        self.frameskip_period = max(1, int(frameskip_period))

        # 统计
        self.frames_total = 0
        self.frames_lander_ok = 0
        self.frames_terrain_ok = 0

        # 历史状态
        self.prev_theta = None
        self.prev_cx = None
        self.prev_cy = None
        self.prev_time = None
        self.last_state = None

        # 地形与接地检测
        self.terrain = TerrainTracker(update_period=3, canny1=50, canny2=120, smooth_kernel=9)
        self.gap_on   = 4.0   # px，进入接触的最大间隙
        self.gap_off  = 8.0   # px，脱离接触的最小间隙
        self.vy_tol   = 0.02  # 归一 vy 阈值
        self.on_need  = 2     # 连续几帧满足才置1
        self.off_need = 2     # 连续几帧满足才置0
        self._cntL_on = self._cntL_off = 0
        self._cntR_on = self._cntR_off = 0
        self._legL = 0.0
        self._legR = 0.0

        # 角度约束统计
        self.hard_rejects = 0
        self.soft_clamps  = 0

    # 可由上层在 episode 开头调用
    def begin_episode(self):
        self.prev_theta = None
        self.prev_cx = None
        self.prev_cy = None
        self.prev_time = None
        self.last_state = None
        self._cntL_on = self._cntL_off = 0
        self._cntR_on = self._cntR_off = 0
        self._legL = 0.0
        self._legR = 0.0
        self.hard_rejects = 0
        self.soft_clamps = 0

    def reset_history(self):
        self.begin_episode()

    def reset_counters(self):
        self.frames_total = 0
        self.frames_lander_ok = 0
        self.frames_terrain_ok = 0

    # ---- 内部：从 OBB+Pose 推出 θ / 中心点 + 脚尖 ----
    def _infer_once(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # OBB
        r = self.obb(frame_bgr, imgsz=self.imgsz_obb, conf=self.conf_obb, verbose=False)[0]
        has_obb = False
        cx = cy = w = h = theta_obb = None
        if r.obb is not None and r.obb.xywhr is not None and len(r.obb.xywhr):
            idx = int(r.obb.conf.argmax().cpu().item())
            cx, cy, w, h, theta_obb = r.obb.xywhr[idx].cpu().numpy().tolist()
            has_obb = True
        elif r.boxes is not None and len(r.boxes):
            idx = int(r.boxes.conf.argmax().cpu().item())
            x, y, w, h = r.boxes.xywh[idx].cpu().tolist()
            cx, cy, theta_obb = x, y, 0.0
            has_obb = True

        if not has_obb:
            return None

        # 自适应 pad
        pad_local = self.base_pad
        if theta_obb is not None and near_vertical(theta_obb, 20.0):
            pad_local = max(pad_local, 0.30)

        # 裁剪 + Pose
        crop_bgr, meta = affine_and_crop_bgr(frame_rgb, cx, cy, w, h, theta_obb, pad_ratio=pad_local)
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        pr = self.pose(crop_bgr, imgsz=self.imgsz_pose, conf=self.conf_pose, verbose=False)[0]
        ok = (pr.boxes is not None and len(pr.boxes) > 0) and \
             (pr.keypoints is not None) and (getattr(pr.keypoints, "xy", None) is not None) and \
             (len(pr.keypoints.xy) > 0)
        if not ok:
            theta = theta_obb
            return {"cx": cx, "cy": cy, "theta": theta, "pose_ok": False, "H": H, "W": W, "feet": None}

        det_idx = int(pr.boxes.conf.argmax().cpu().item())
        kp_list = pr.keypoints.xy
        if det_idx >= len(kp_list):
            det_idx = 0
        k = kp_list[det_idx].cpu().numpy()
        if k.ndim != 2 or k.shape[0] < 4 or not np.all(np.isfinite(k)):
            return {"cx": cx, "cy": cy, "theta": theta_obb, "pose_ok": False, "H": H, "W": W, "feet": None}

        # 映射回原图
        M, x0c, y0c = meta
        k_rot = k.copy()
        k_rot[:, 0] += x0c
        k_rot[:, 1] += y0c
        inv_M = cv2.invertAffineTransform(M)
        k_homo = np.hstack([k_rot, np.ones((k_rot.shape[0], 1), dtype=np.float32)])
        k_org = (inv_M @ k_homo.T).T  # (K,2)

        theta_pose = robust_angle_from_kpts(k_org, use_pca_fallback=True, min_dx_px=3.0)
        if theta_pose is not None:
            if ang_axis_diff(theta_pose, theta_obb) > self.gate_rad:
                theta = theta_obb
                pose_ok = False
            else:
                theta = theta_pose
                pose_ok = True
        else:
            theta = theta_obb
            pose_ok = False

        feet = None
        if k_org.shape[0] >= 4 and np.all(np.isfinite(k_org[:4])):
            feet = k_org[2:4, :2].copy()  # (2,2)

        return {"cx": cx, "cy": cy, "theta": theta, "pose_ok": pose_ok, "H": H, "W": W, "feet": feet}

    # ---- 角度变化双约束 & 平滑 ----
    def _apply_angle_limits(self, theta_candidate, cx, cy, now_ts):
        """
        对候选角度进行软/硬约束与平滑：
        - 计算 dt 与像素速度 v_pix
        - 软上限：ω_soft = soft_floor + soft_gain * v_pix （deg/s）
        - 硬上限：max_spin_hard_deg_s
        - 超硬限 → reject：用上次角速度（若有）在硬限内外推
        - 超软限 → clamp：把角速度夹到 ±ω_soft
        - 平滑：theta = alpha*theta_prev + (1-alpha)*theta_new
        """
        if self.prev_theta is None or self.prev_time is None or self.prev_cx is None:
            return theta_candidate  # 第一帧接受

        dt = max(1e-3, now_ts - self.prev_time)
        # 像素速度
        dx = (cx - self.prev_cx)
        dy = (cy - self.prev_cy)
        v_pix = math.hypot(dx, dy) / dt  # px/s

        # 计算候选角速度（deg/s）
        dtheta = ang_diff(theta_candidate, self.prev_theta)
        omega_cand_deg_s = math.degrees(dtheta) / dt

        # 软上限（动态）
        omega_soft = self.soft_floor_deg_s + self.soft_gain_deg_per_pix_s * v_pix

        # 先硬限判定
        if abs(omega_cand_deg_s) > self.max_spin_hard_deg_s:
            # 拒绝：用“可信角速度”外推。可信角速度=把候选角速度夹紧到硬限（或 0）
            self.hard_rejects += 1
            omega_trust = float(np.clip(omega_cand_deg_s, -self.max_spin_hard_deg_s, self.max_spin_hard_deg_s))
            theta_new = self.prev_theta + math.radians(omega_trust * dt)
        else:
            # 软限夹紧
            if abs(omega_cand_deg_s) > omega_soft:
                self.soft_clamps += 1
            omega_clamped = float(np.clip(omega_cand_deg_s, -omega_soft, omega_soft))
            theta_new = self.prev_theta + math.radians(omega_clamped * dt)

        # 平滑
        theta_smooth = wrap_pi(self.alpha * self.prev_theta + (1.0 - self.alpha) * theta_new)
        return theta_smooth

    # ---- 打包 8 维状态 ----
    def _pack_state(self, theta, cx, cy, W, H, now_ts, legL, legR):
        x_norm = (cx - W/2) / (W/2)
        y_norm = (cy - H/2) / (H/2)

        vx_norm = vy_norm = 0.0
        if (self.prev_cx is not None) and (self.prev_cy is not None) and (self.prev_time is not None):
            dt = max(1e-3, now_ts - self.prev_time)
            vx_norm = ((cx - self.prev_cx) / (W/2)) / dt
            vy_norm = ((cy - self.prev_cy) / (H/2)) / dt

        cos_t = math.cos(theta); sin_t = math.sin(theta)
        z = np.array([cos_t, sin_t, x_norm, y_norm, vx_norm, vy_norm, legL, legR], dtype=np.float32)

        self.prev_theta = theta
        self.prev_cx = cx
        self.prev_cy = cy
        self.prev_time = now_ts
        self.last_state = z
        return z

    # ---- 全推理 ----
    def update_full(self, frame_bgr):
        self.frames_total += 1
        now = time.perf_counter()

        out = self._infer_once(frame_bgr)
        if out is None:
            self.last_state = None
            return None

        cx, cy, theta_raw = out["cx"], out["cy"], out["theta"]
        W, H = out["W"], out["H"]
        feet = out.get("feet", None)

        # 角度双约束 + 平滑
        theta = self._apply_angle_limits(theta_raw, cx, cy, now)

        # 更新地形高度场（周期性）
        terrain_y = self.terrain.update(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        # 接地状态机（若有脚尖）
        if feet is not None and np.all(np.isfinite(feet)):
            idx = np.argsort(feet[:,0])
            lf, rf = feet[idx[0]], feet[idx[1]]

            yL_ground = self.terrain.query(lf[0])
            yR_ground = self.terrain.query(rf[0])
            gapL = yL_ground - lf[1]   # y向下
            gapR = yR_ground - rf[1]

            # 估计当前 vy_norm（用于接地阈值）
            vy_norm = 0.0
            if (self.prev_cy is not None) and (self.prev_time is not None):
                dt_v = max(1e-3, now - self.prev_time)
                vy_norm = ((cy - self.prev_cy) / (H/2)) / dt_v

            # 左脚
            if (0.0 <= gapL <= self.gap_on) and (abs(vy_norm) <= self.vy_tol):
                self._cntL_on += 1; self._cntL_off = 0
                if self._cntL_on >= self.on_need: self._legL = 1.0
            else:
                if (gapL >= self.gap_off) or (abs(vy_norm) > self.vy_tol):
                    self._cntL_off += 1; self._cntL_on = 0
                    if self._cntL_off >= self.off_need: self._legL = 0.0

            # 右脚
            if (0.0 <= gapR <= self.gap_on) and (abs(vy_norm) <= self.vy_tol):
                self._cntR_on += 1; self._cntR_off = 0
                if self._cntR_on >= self.on_need: self._legR = 1.0
            else:
                if (gapR >= self.gap_off) or (abs(vy_norm) > self.vy_tol):
                    self._cntR_off += 1; self._cntR_on = 0
                    if self._cntR_off >= self.off_need: self._legR = 0.0

        # 统计
        if out.get("pose_ok", False):
            self.frames_lander_ok += 1
            self.frames_terrain_ok += 1

        return self._pack_state(theta, cx, cy, W, H, now_ts=now, legL=self._legL, legR=self._legR)

    # ---- 仅预测（frame-skip）----
    def predict_only(self):
        if self.last_state is None or self.prev_time is None or self.prev_cx is None:
            return None

        now = time.perf_counter()
        dt = max(1e-3, now - self.prev_time)

        cos_t, sin_t, x_norm, y_norm, vx_n, vy_n, legL, legR = self.last_state.tolist()
        x_norm_pred = float(np.clip(x_norm + vx_n * dt, -1.5, 1.5))
        y_norm_pred = float(np.clip(y_norm + vy_n * dt, -1.5, 1.5))

        z = np.array([cos_t, sin_t, x_norm_pred, y_norm_pred, vx_n, vy_n, legL, legR], dtype=np.float32)
        self.last_state = z
        return z

    # ---- 调参/诊断 ----
    def get_debug_stats(self):
        return {
            "hard_rejects": int(self.hard_rejects),
            "soft_clamps": int(self.soft_clamps),
            "frames_total": int(self.frames_total),
            "lander_ok": int(self.frames_lander_ok),
            "terrain_ok": int(self.frames_terrain_ok)
        }
