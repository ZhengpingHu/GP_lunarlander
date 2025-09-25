#!/usr/bin/env python3
# yolo_state_4d.py
# -*- coding: utf-8 -*-
"""
YoloStateEstimator4D
- 双模型：YOLOv8-OBB（定位+大致朝向）+ YOLOv8-Pose（关键点角度，脚尖连线 + PCA 兜底）
- 输出4维状态: [cosθ, sinθ, x_norm, y_norm]
- begin_episode(): 清空历史（服务端启动时会调用）
- update_full(frame_bgr): 跑一帧完整推理，返回 4D 状态（numpy.float32）
- predict_only(): 不跑YOLO，返回上一次的 4D 状态（若无则 None）
"""

from typing import Optional, Tuple
import math
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(f"Ultralytics 未安装或导入失败: {e}")


# ----------------- 角度/工具函数 -----------------
def wrap_pi(a: float) -> float:
    """wrap angle to (-pi, pi]"""
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


def ang_diff(a: float, b: float) -> float:
    """|a - b| wrapped"""
    return abs(wrap_pi(a - b))


def ang_axis_diff(a: float, b: float) -> float:
    """轴向无向差：考虑 180° 对称"""
    d = ang_diff(a, b)
    return min(d, abs(wrap_pi(d - math.pi)))


def near_vertical(theta: float, thresh_deg: float = 20.0) -> bool:
    """是否接近竖直（±90°）"""
    d = min(abs(wrap_pi(theta - math.pi / 2)), abs(wrap_pi(theta + math.pi / 2)))
    return d < math.radians(thresh_deg)


def robust_angle_from_kpts(k_org: np.ndarray,
                           use_pca_fallback: bool = True,
                           min_dx_px: float = 3.0) -> Optional[float]:
    """
    用关键点求角度（优先脚尖连线，dx太小用PCA兜底）
    约定：k_org 至少前4点分别为 [*, *, left_foot, right_foot]
    返回：theta (rad) or None
    """
    if k_org is None or k_org.ndim != 2 or k_org.shape[0] < 4:
        return None
    if not np.all(np.isfinite(k_org[:4, :2])):
        return None

    feet = k_org[2:4, :2].astype(np.float32).copy()
    order = np.argsort(feet[:, 0])
    left_foot, right_foot = feet[order[0]], feet[order[1]]

    dx = float(right_foot[0] - left_foot[0])
    dy = float(right_foot[1] - left_foot[1])

    # 足够水平分离 → 用脚尖连线
    if abs(dx) >= min_dx_px:
        return wrap_pi(math.atan2(dy, dx))

    if not use_pca_fallback:
        return wrap_pi(math.atan2(dy, dx))

    # PCA 兜底，使用前4点
    P = k_org[:4, :2].astype(np.float32)
    if not np.all(np.isfinite(P)):
        return wrap_pi(math.atan2(dy, dx))

    mu = P.mean(axis=0, keepdims=True)
    X = P - mu
    C = (X.T @ X) / max(len(P) - 1, 1)
    eigvals, eigvecs = np.linalg.eig(C)
    v = eigvecs[:, int(np.argmax(eigvals))]

    leftmost = P[np.argmin(P[:, 0])]
    rightmost = P[np.argmax(P[:, 0])]
    ref = (rightmost - leftmost).astype(np.float32)
    if np.dot(v, ref) < 0:
        v = -v

    theta = math.atan2(float(v[1]), float(v[0]))
    return wrap_pi(theta)


# ----------------- 轴对齐裁剪 -----------------
def aa_crop(frame_bgr: np.ndarray, cx: float, cy: float, w: float, h: float, pad: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    轴对齐裁剪（不旋转），返回 (crop_bgr, (x0, y0))
    """
    H, W = frame_bgr.shape[:2]
    wp = w * (1.0 + 2.0 * pad)
    hp = h * (1.0 + 2.0 * pad)
    x0 = int(round(cx - wp / 2))
    y0 = int(round(cy - hp / 2))
    x1 = int(round(cx + wp / 2))
    y1 = int(round(cy + hp / 2))

    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x1), min(H, y1)
    if x1c <= x0c or y1c <= y0c:
        return None, (0, 0)
    return frame_bgr[y0c:y1c, x0c:x1c].copy(), (x0c, y0c)


# ==========================================================
#                       主类
# ==========================================================
class YoloStateEstimator4D:
    """
    4D 状态估计器（OBB + Pose）
    - 输出: [cosθ, sinθ, x_norm, y_norm]
    - x_norm, y_norm: 以画面中心为原点，范围约 [-1, 1]
    """

    def __init__(self,
                 obb_model_path: str,
                 pose_model_path: str,
                 device: str = "cuda:0",
                 imgsz_obb: int = 640,
                 imgsz_pose: int = 384,
                 conf_obb: float = 0.25,
                 conf_pose: float = 0.20,
                 base_pad: float = 0.25,
                 gate_deg: float = 120.0,
                 smooth_alpha: float = 0.2):
        # 模型
        self.obb = YOLO(obb_model_path, task="obb")
        self.pose = YOLO(pose_model_path, task="pose")

        # 推理参数
        self.device = device
        self.imgsz_obb = int(imgsz_obb)
        self.imgsz_pose = int(imgsz_pose)
        self.conf_obb = float(conf_obb)
        self.conf_pose = float(conf_pose)
        self.base_pad = float(base_pad)
        self.gate_rad = math.radians(float(gate_deg))
        self.alpha = float(np.clip(smooth_alpha, 0.0, 1.0))

        # 预测 kwargs（Ultralytics 统一走 predictor）
        self._pred_obb = dict(imgsz=self.imgsz_obb, device=self.device, conf=self.conf_obb, verbose=False)
        self._pred_pose = dict(imgsz=self.imgsz_pose, device=self.device, conf=self.conf_pose, verbose=False)

        # 历史/状态
        self.theta_smooth: Optional[float] = None  # EMA 平滑的θ
        self.last_state: Optional[np.ndarray] = None  # 4D
        self.last_cxy: Optional[Tuple[float, float]] = None
        self.last_HW: Optional[Tuple[int, int]] = None

    # ===== 生命周期 =====
    def begin_episode(self):
        """新 episode 开始前调用，清空历史"""
        self.theta_smooth = None
        self.last_state = None
        self.last_cxy = None
        self.last_HW = None

    def reset_history(self):
        """兼容旧命名"""
        self.begin_episode()

    # ===== 仅预测 =====
    def predict_only(self) -> Optional[np.ndarray]:
        """
        不做YOLO，直接返回上一帧 4D 状态；若无则 None
        """
        return None if self.last_state is None else self.last_state.copy()

    # ===== 全推理 =====
    def update_full(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        跑一帧完整推理：OBB → AABB裁剪 → Pose → 角度融合 → 打包 4D
        """
        out = self._infer_once(frame_bgr)
        if out is None:
            self.last_state = None
            return None

        cx, cy, theta, W, H = out["cx"], out["cy"], out["theta"], out["W"], out["H"]

        # 归一化坐标
        x_norm = (cx - W / 2) / (W / 2)
        y_norm = (cy - H / 2) / (H / 2)

        # 角度 EMA 平滑
        if self.theta_smooth is None:
            self.theta_smooth = theta
        else:
            # 指定在环形域上平滑：把旧值旋到邻近新值再插值
            # 使得平滑不过界
            d = wrap_pi(theta - self.theta_smooth)
            self.theta_smooth = wrap_pi(self.theta_smooth + self.alpha * d)

        cos_t, sin_t = math.cos(self.theta_smooth), math.sin(self.theta_smooth)
        z = np.array([cos_t, sin_t, x_norm, y_norm], dtype=np.float32)

        # 存历史
        self.last_state = z
        self.last_cxy = (cx, cy)
        self.last_HW = (H, W)
        return z

    # ===== 内部：一次推理 =====
    def _infer_once(self, frame_bgr: np.ndarray) -> Optional[dict]:
        H, W = frame_bgr.shape[:2]

        # ---------- OBB ----------
        r = self.obb(frame_bgr, **self._pred_obb)[0]
        has_obb = False
        cx = cy = w = h = None
        theta_obb = None

        if (getattr(r, "obb", None) is not None and
                getattr(r.obb, "xywhr", None) is not None and
                len(r.obb.xywhr) > 0):
            idx = int(r.obb.conf.argmax().item())
            cx, cy, w, h, theta_obb = r.obb.xywhr[idx].cpu().numpy().tolist()
            has_obb = True
        elif (getattr(r, "boxes", None) is not None and len(r.boxes) > 0):
            idx = int(r.boxes.conf.argmax().item())
            cx, cy, w, h = r.boxes.xywh[idx].cpu().numpy().tolist()
            theta_obb = 0.0
            has_obb = True

        if not has_obb:
            return None

        # ---------- 裁剪（轴对齐） ----------
        pad_local = self.base_pad
        if theta_obb is not None and near_vertical(theta_obb, 20.0):
            pad_local = max(pad_local, 0.30)

        crop_bgr, (x0c, y0c) = aa_crop(frame_bgr, cx, cy, w, h, pad_local)
        if crop_bgr is None or crop_bgr.size == 0:
            # 退化：直接用 OBB 中心 + 角
            theta_final = theta_obb if theta_obb is not None else 0.0
            return {"cx": cx, "cy": cy, "theta": theta_final, "W": W, "H": H}

        # ---------- Pose ----------
        pr = self.pose(crop_bgr, **self._pred_pose)[0]
        pose_ok = (getattr(pr, "boxes", None) is not None and len(pr.boxes) > 0 and
                   getattr(pr, "keypoints", None) is not None and
                   getattr(pr.keypoints, "xy", None) is not None and
                   len(pr.keypoints.xy) > 0)

        if not pose_ok:
            theta_final = theta_obb if theta_obb is not None else 0.0
            return {"cx": cx, "cy": cy, "theta": theta_final, "W": W, "H": H}

        det_idx = int(pr.boxes.conf.argmax().item())
        kp = pr.keypoints.xy[det_idx].cpu().numpy()
        if kp.ndim != 2 or kp.shape[0] < 4 or not np.all(np.isfinite(kp)):
            theta_final = theta_obb if theta_obb is not None else 0.0
            return {"cx": cx, "cy": cy, "theta": theta_final, "W": W, "H": H}

        # 映射回原图（轴对齐裁剪：加上偏移即可）
        kp_org = kp.copy()
        kp_org[:, 0] += x0c
        kp_org[:, 1] += y0c

        theta_pose = robust_angle_from_kpts(kp_org, use_pca_fallback=True, min_dx_px=3.0)

        # ---------- 角度融合 ----------
        if theta_pose is None and theta_obb is None:
            theta_final = 0.0
        elif theta_pose is None:
            theta_final = theta_obb
        elif theta_obb is None:
            theta_final = theta_pose
        else:
            # 轴向无向门限（防翻转）：差太大就用 OBB
            if ang_axis_diff(theta_pose, theta_obb) > self.gate_rad:
                theta_final = theta_obb
            else:
                theta_final = theta_pose

        return {"cx": cx, "cy": cy, "theta": theta_final, "W": W, "H": H}
