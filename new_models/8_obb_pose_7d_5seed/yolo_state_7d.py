#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import time
from typing import Optional, Tuple

import numpy as np
import torch

# 建议限线程，减少与多进程/BLAS 冲突
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(
        f"Import ultralytics failed: {e}\n"
        "Please `pip install ultralytics` and ensure your OBB/Pose weights are compatible."
    )

# --------- 小工具 ----------
def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def ang_diff(a: float, b: float) -> float:
    return wrap_pi(a - b)

def clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

# ==========================================================
# 7D 本地状态器（不经 RPC）
# 输出 z = [cx, cy, theta, vx, vy, vtheta, speed]
#   - cx, cy ∈ [0,1] 图像归一化坐标
#   - theta 弧度，wrap 到 [-pi, pi)
#   - 速度单位：vx,vy 为归一化坐标/秒；vtheta 为弧度/秒；speed=√(vx²+vy²)
# 稳定策略：
#   * EMA 平滑（位置与角）
#   * 角度软/硬限制（软限随 |vtheta_prev| 动态变化）
#   * OBB/POSE 融合（根据上一次角度符号进行轻微偏置）
#   * 动作一致性门控：上一动作与测得角速度的期望符号不符时，向预测收缩
# ==========================================================
class YoloStateEstimator7D:
    def __init__(
        self,
        obb_model_path: str,
        pose_model_path: Optional[str] = None,
        device: str = "cuda:0",
        imgsz_obb: int = 640,
        imgsz_pose: int = 384,
        conf_obb: float = 0.25,
        conf_pose: float = 0.20,
        iou_obb: float = 0.7,
        iou_pose: float = 0.7,
        max_det: int = 10,
        agnostic_nms: bool = True,
        pad: float = 0.25,
        smooth_alpha: float = 0.20,      # EMA 系数
        gate_soft_k: float = 0.8,        # 软限权重（随上次角速度线性）
        gate_soft_bias_deg: float = 10.0,# 软限基数（度）
        gate_hard_deg: float = 180.0,    # 角度硬限（度）
        omega_hard_deg: float = 360.0,   # 角速度硬限（度/秒），用于异常判定
        fps_hint: float = 30.0           # 用于估计 dt（若没有真实时间间隔）
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.obb = YOLO(obb_model_path).to(self.device)
        self.pose = None
        if pose_model_path and len(str(pose_model_path).strip()) > 0:
            self.pose = YOLO(pose_model_path).to(self.device)

        # 推理参数
        self.conf_obb = conf_obb
        self.conf_pose = conf_pose
        self.iou_obb = iou_obb
        self.iou_pose = iou_pose
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        self.imgsz_obb = imgsz_obb
        self.imgsz_pose = imgsz_pose
        self.pad = pad

        # 门控/平滑
        self.alpha = float(smooth_alpha)
        self.gate_soft_k = float(gate_soft_k)
        self.gate_soft_bias = math.radians(gate_soft_bias_deg)
        self.gate_hard = math.radians(gate_hard_deg)
        self.omega_hard = math.radians(omega_hard_deg)

        # 时间
        self.dt_default = 1.0 / max(1.0, fps_hint)
        self.t_last: Optional[float] = None

        # 状态缓存
        self._z: Optional[np.ndarray] = None      # 当前 7D
        self._z_prior: Optional[np.ndarray] = None# 上次可信 [cx,cy,theta]
        self._last_action: Optional[int] = None   # 0=无喷,1=左姿态,2=主喷,3=右姿态

    # ---------- 外部接口 ----------
    def begin_episode(self):
        self._z = None
        self._z_prior = None
        self._last_action = None
        self.t_last = None

    def notify_action(self, act: int):
        """在 env.step(act) 后调用，更新上一动作，用于动作一致性门控。"""
        self._last_action = int(act)

    # ---------- 内部推理 ----------
    @torch.no_grad()
    def _infer_obb(self, frame_bgr: np.ndarray) -> Optional[Tuple[float, float, float]]:
        res = self.obb.predict(
            frame_bgr,
            imgsz=self.imgsz_obb,
            conf=self.conf_obb,
            iou=self.iou_obb,
            max_det=self.max_det,
            agnostic_nms=self.agnostic_nms,
            half=False,
            device=self.device,
            verbose=False
        )
        if len(res) == 0:
            return None
        r = res[0]
        if not hasattr(r, "obb") or r.obb is None or len(r.obb) == 0:
            return None

        boxes = r.obb
        # 最高置信
        if hasattr(r, "boxes") and r.boxes is not None and r.boxes.conf is not None:
            scores = r.boxes.conf.detach().cpu().numpy()
            idx = int(np.argmax(scores))
        else:
            idx = 0

        # 中心归一化坐标
        if hasattr(boxes, "xywhn"):
            xywhn = boxes.xywhn.detach().cpu().numpy()[idx]
            cx, cy = float(xywhn[0]), float(xywhn[1])
        else:
            xywh = boxes.xywh.detach().cpu().numpy()[idx]
            H, W = frame_bgr.shape[:2]
            cx, cy = float(xywh[0] / W), float(xywh[1] / H)

        # 角度
        if hasattr(boxes, "theta"):
            th = float(boxes.theta.detach().cpu().numpy()[idx])
        else:
            th = 0.0
        th = wrap_pi(th)
        return (cx, cy, th)

    @torch.no_grad()
    def _infer_pose_theta(self, frame_bgr: np.ndarray) -> Optional[float]:
        if self.pose is None:
            return None
        res = self.pose.predict(
            frame_bgr,
            imgsz=self.imgsz_pose,
            conf=self.conf_pose,
            iou=self.iou_pose,
            max_det=self.max_det,
            agnostic_nms=self.agnostic_nms,
            half=False,
            device=self.device,
            verbose=False
        )
        if len(res) == 0:
            return None
        r = res[0]
        if not hasattr(r, "keypoints") or r.keypoints is None or len(r.keypoints) == 0:
            return None

        kp = r.keypoints
        if hasattr(r, "boxes") and r.boxes is not None and r.boxes.conf is not None:
            scores = r.boxes.conf.detach().cpu().numpy()
            idx = int(np.argmax(scores))
        else:
            idx = 0

        if hasattr(kp, "xyn"):
            kxy = kp.xyn[idx].detach().cpu().numpy()
        else:
            kxy = kp.xy[idx].detach().cpu().numpy()
        if kxy.shape[0] < 2:
            return None

        p0, p1 = kxy[0], kxy[1]
        dx, dy = float(p1[0] - p0[0]), float(p1[1] - p0[1])
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        theta = math.atan2(dy, dx)
        return wrap_pi(theta)

    def _fuse_theta(self, th_obb: Optional[float], th_pose: Optional[float], dt: float) -> Optional[float]:
        if th_obb is None and th_pose is None:
            return None
        if th_pose is None:
            return th_obb
        if th_obb is None:
            return th_pose

        theta_prior = self._z_prior[2] if self._z_prior is not None else th_obb
        w_hi, w_lo = 0.85, 0.15
        w_obb, w_pose = (w_hi, w_lo) if theta_prior >= 0 else (w_lo, w_hi)

        dlim = min(self.gate_hard, self.omega_hard * dt)
        d1 = clip(ang_diff(th_obb, theta_prior), -dlim, +dlim)
        d2 = clip(ang_diff(th_pose, theta_prior), -dlim, +dlim)
        return wrap_pi(theta_prior + (w_obb * d1 + w_pose * d2) / (w_obb + w_pose))

    def _apply_action_consistency(self, th_used: float, th_pred: float, dt: float) -> float:
        if self._last_action is None or self._z is None or dt <= 1e-6:
            return th_used

        # 1=左姿态 → 期望顺时针 vtheta<0；3=右姿态 → 期望逆时针 vtheta>0；2=主喷 → 不强约束
        exp = 0
        if self._last_action == 1:
            exp = -1
        elif self._last_action == 3:
            exp = +1
        else:
            exp = 0

        if exp == 0:
            return th_used

        vtheta_meas = ang_diff(th_used, float(self._z[2])) / dt
        score = exp * vtheta_meas  # 希望 >0
        tau_incon = 0.25 * self.omega_hard
        if score < 0:
            beta = max(0.0, 1.0 + score / tau_incon)  # -> [0,1]
            d_adj = ang_diff(th_used, th_pred) * beta
            th_used = wrap_pi(th_pred + d_adj)
            if (abs(vtheta_meas) > self.omega_hard) and (score < -tau_incon):
                th_used = th_pred
        return th_used

    def _gate_and_smooth(self, cx1: float, cy1: float, th1: float, dt: float) -> Tuple[float, float, float]:
        if self._z_prior is None:
            cx, cy, th = cx1, cy1, th1
        else:
            cx0, cy0, th0 = float(self._z_prior[0]), float(self._z_prior[1]), float(self._z_prior[2])

            vtheta_prev = 0.0 if self._z is None else float(self._z[5])
            soft_limit = min(self.gate_hard, self.gate_soft_bias + self.gate_soft_k * abs(vtheta_prev) * dt)
            dth = clip(ang_diff(th1, th0), -self.gate_hard, self.gate_hard)
            dth = clip(dth, -soft_limit, soft_limit)
            th_g = wrap_pi(th0 + dth)

            # EMA
            cx = self.alpha * cx1 + (1.0 - self.alpha) * cx0
            cy = self.alpha * cy1 + (1.0 - self.alpha) * cy0
            th = wrap_pi(th0 + self.alpha * ang_diff(th_g, th0))

        self._z_prior = np.array([cx, cy, th], dtype=np.float32)
        return cx, cy, th

    # ---------- 外部：完整观测 ----------
    def update_full(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        t = time.time()
        dt = self.dt_default if self.t_last is None else max(1e-3, t - self.t_last)
        self.t_last = t

        obb = self._infer_obb(frame_bgr)
        if obb is None:
            return None
        cx_m, cy_m, th_obb = obb
        th_pose = self._infer_pose_theta(frame_bgr)

        th_pred = wrap_pi(float(self._z[2]) + float(self._z[5]) * dt) if self._z is not None else th_obb
        th_m = self._fuse_theta(th_obb, th_pose, dt) or th_pred

        cx_s, cy_s, th_s = self._gate_and_smooth(cx_m, cy_m, th_m, dt)
        th_s = self._apply_action_consistency(th_s, th_pred, dt)

        if self._z is None:
            vx = vy = vtheta = 0.0
        else:
            vx = (cx_s - float(self._z[0])) / dt
            vy = (cy_s - float(self._z[1])) / dt
            vtheta = ang_diff(th_s, float(self._z[2])) / dt

        speed = math.hypot(vx, vy)
        self._z = np.array([cx_s, cy_s, th_s, vx, vy, vtheta, speed], dtype=np.float32)
        return self._z.copy()

    # ---------- 外部：预测外推 ----------
    def predict_only(self) -> Optional[np.ndarray]:
        if self._z is None:
            return None
        t = time.time()
        dt = self.dt_default if self.t_last is None else max(1e-3, t - self.t_last)
        self.t_last = t

        cx, cy, th, vx, vy, vtheta, _ = [float(x) for x in self._z]
        cx2 = cx + vx * dt
        cy2 = cy + vy * dt
        th2 = wrap_pi(th + vtheta * dt)
        speed2 = math.hypot(vx, vy)
        self._z = np.array([cx2, cy2, th2, vx, vy, vtheta, speed2], dtype=np.float32)
        return self._z.copy()
