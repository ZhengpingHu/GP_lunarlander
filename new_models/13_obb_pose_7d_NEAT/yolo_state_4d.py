# yolo_state_7d.py
# -*- coding: utf-8 -*-

import time
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    # 尽量兼容：既支持传入 ultralytics 的模型实例，也支持你后面改成别的推理封装
    from ultralytics import YOLO
except Exception:
    YOLO = None


def _rad_wrap(a: float) -> float:
    """wrap angle to [-pi, pi)."""
    a = (a + math.pi) % (2.0 * math.pi) - math.pi
    return a


@dataclass
class FilterParams:
    # EMA 平滑
    alpha_pos: float = 0.2     # x/y/theta 平滑系数
    alpha_vel: float = 0.2     # vx/vy/vtheta 平滑系数

    # 门控与物理约束
    k_soft: float = 1.5        # 软上限系数（基于 |vtheta| 的放大）
    base_margin_deg: float = 5.0  # 软上限的基础角度余量（度）
    omega_max_deg: float = 180.0  # 硬上限：单侧全推的最大角速度（度/秒）

    # 其他
    min_dt: float = 1.0 / 120.0   # 防止 dt 极小
    max_dt: float = 0.25          # 防止长时间卡顿导致 dt 过大
    conf_obb: float = 0.25        # OBB 置信度阈值
    imgsz_obb: int = 640
    pad: float = 0.25
    gate_deg: float = 120.0       # 角度观测允许的最大相对差（大闸门，超过直接拒绝）


class YoloStateEstimator7D:
    """
    - 仅依赖 OBB：输出 [x, y, theta, vx, vy, vtheta, speed] (float32)
    - 提供 begin_episode / update_full / predict_only 三个方法（方便直接替换你现有的 estimator）
    - 输入图像为 BGR (np.uint8, HxWx3)
    """

    def __init__(
        self,
        obb_model: "YOLO",
        device: str = "cuda:0",
        params: FilterParams = FilterParams(),
        class_id: int = 0,          # lander 的类别 id（如果你的权重里只有一个类，填 0）
        w: Optional[int] = None,    # 可选：如果你愿意提前告诉分辨率，可以加速归一化
        h: Optional[int] = None,
    ):
        self.device = device
        self.params = params
        self.class_id = class_id

        # 模型：传实例（推荐）。若传的是路径，自动加载。
        if hasattr(obb_model, "predict") or hasattr(obb_model, "__call__"):
            self.obb = obb_model
        else:
            if YOLO is None:
                raise RuntimeError("ultralytics 未安装，且未提供可调用的 obb_model 实例。")
            self.obb = YOLO(obb_model)

        # 运行时缓存
        self._t_prev: Optional[float] = None
        self._z: Optional[np.ndarray] = None         # 上一帧的平滑状态 (7,)
        self._z_prior: Optional[np.ndarray] = None   # 预测状态 (7,)
        self._w = w
        self._h = h

        # 预计算常数
        self._base_margin = math.radians(self.params.base_margin_deg)
        self._omega_max = math.radians(self.params.omega_max_deg)
        self._gate_rad = math.radians(self.params.gate_deg)

    # --------- 生命周期 ---------
    def begin_episode(self):
        self._t_prev = None
        self._z = None
        self._z_prior = None

    reset = begin_episode  # 兼容你之前的命名

    # --------- 主调用：完整更新（推理 + 滤波） ---------
    def update_full(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        完整帧更新：做一次 OBB 推理 -> 获得测量 -> 物理预测 -> 门控/平滑 -> 输出 7D
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        t_now = time.perf_counter()
        dt = self._estimate_dt(t_now)

        # 1) 测量：从 OBB 提取 (x, y, theta)（归一化坐标 & 弧度）
        meas = self._measure_from_obb(frame_bgr)
        if meas is None:
            # 拒绝观测，退回预测
            z = self._predict_only(dt)
            return z

        x_m, y_m, th_m = meas

        # 2) 预测：常速度模型
        z_prior = self._predict_only(dt)  # 里面会用到 self._z
        # 若首帧：直接用测量初始化
        if z_prior is None:
            self._z = np.array([x_m, y_m, th_m, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self._t_prev = t_now
            return self._z.copy()

        # 3) 门控：角度观测相对预测的增量
        th_pred = z_prior[2]
        dtheta = _rad_wrap(th_m - th_pred)

        soft_limit = self._soft_angle_limit(abs(z_prior[5]), dt)  # 基于当前角速度的软上限
        hard_limit = self._omega_max * dt                          # 硬上限

        accept_angle = True
        th_used = th_m

        # 大闸门（极端离谱，直接拒绝）
        if abs(dtheta) > self._gate_rad:
            accept_angle = False
            th_used = th_pred
        else:
            # 软上限：超了就“拉回”靠近预测
            if abs(dtheta) > soft_limit:
                th_used = th_pred + np.clip(dtheta, -soft_limit, soft_limit)
            # 硬上限：最终还是不能超过硬上限
            dtheta2 = _rad_wrap(th_used - th_pred)
            if abs(dtheta2) > hard_limit:
                accept_angle = False
                th_used = th_pred + np.clip(dtheta2, -hard_limit, hard_limit)

        th_used = _rad_wrap(th_used)

        # 4) 位置门控（可选：一般 OBB 中心位置不会跳太离谱，先使用 EMA 即可）
        x_used = x_m
        y_used = y_m

        # 5) 速度测量（用“使用后的姿态/位置” 与 上一帧平滑值差分）
        vx_m, vy_m, vth_m = 0.0, 0.0, 0.0
        if self._z is not None and dt > 1e-6:
            vx_m = (x_used - self._z[0]) / dt
            vy_m = (y_used - self._z[1]) / dt
            vth_m = _rad_wrap(th_used - self._z[2]) / dt

        # 6) EMA 平滑（若拒绝角度观测，theta/ vtheta 的观测就用预测替代）
        alpha_p = self.params.alpha_pos
        alpha_v = self.params.alpha_vel

        if not accept_angle:
            th_used = th_pred           # 角度回到预测
            vth_m = z_prior[5]          # 角速度也用预测

        x_s = alpha_p * x_used + (1.0 - alpha_p) * z_prior[0]
        y_s = alpha_p * y_used + (1.0 - alpha_p) * z_prior[1]
        th_s = _rad_wrap(alpha_p * th_used + (1.0 - alpha_p) * th_pred)

        vx_s = alpha_v * vx_m + (1.0 - alpha_v) * z_prior[3]
        vy_s = alpha_v * vy_m + (1.0 - alpha_v) * z_prior[4]
        vth_s = alpha_v * vth_m + (1.0 - alpha_v) * z_prior[5]

        speed = float(math.hypot(vx_s, vy_s))

        self._z = np.array([x_s, y_s, th_s, vx_s, vy_s, vth_s, speed], dtype=np.float32)
        self._z_prior = z_prior
        self._t_prev = t_now
        return self._z.copy()

    # --------- 仅预测（不做推理） ----------
    def predict_only(self) -> Optional[np.ndarray]:
        t_now = time.perf_counter()
        dt = self._estimate_dt(t_now)
        z = self._predict_only(dt)
        self._t_prev = t_now
        return None if z is None else z.copy()

    # --------- 内部：常速度预测 ----------
    def _predict_only(self, dt: float) -> Optional[np.ndarray]:
        if self._z is None:
            return None
        dt = float(np.clip(dt, self.params.min_dt, self.params.max_dt))

        x, y, th, vx, vy, vth, _ = [float(v) for v in self._z]
        x_p = x + vx * dt
        y_p = y + vy * dt
        th_p = _rad_wrap(th + vth * dt)

        speed_p = float(math.hypot(vx, vy))
        self._z_prior = np.array([x_p, y_p, th_p, vx, vy, vth, speed_p], dtype=np.float32)
        return self._z_prior

    # --------- 内部：估计 dt ----------
    def _estimate_dt(self, t_now: float) -> float:
        if self._t_prev is None:
            self._t_prev = t_now
            return 1.0 / 60.0  # 初次估一个合理的帧率
        dt = t_now - self._t_prev
        # clamp
        if dt < self.params.min_dt:
            dt = self.params.min_dt
        elif dt > self.params.max_dt:
            dt = self.params.max_dt
        return dt

    # --------- 内部：从 OBB 取测量 ----------
    def _measure_from_obb(self, frame_bgr: np.ndarray) -> Optional[Tuple[float, float, float]]:
        H, W = frame_bgr.shape[:2]
        self._w = self._w or W
        self._h = self._h or H

        # Ultralytics 调用（单帧）
        # 注意：如果你在 RPC 服务里已把模型绑到 GPU，这里就只会是一次前向，不会反复加载
        res = self.obb(
            frame_bgr,
            imgsz=self.params.imgsz_obb,
            conf=self.params.conf_obb,
            verbose=False
        )
        if not res or len(res) == 0:
            return None

        r = res[0]
        if not hasattr(r, "obb") or r.obb is None or len(r.obb) == 0:
            return None

        # 选取最高 score 的一个
        # Ultralytics OBB 的属性在不同版本里名称略有差异：xywhr / xyxyxyxy 等
        # 这里尽量使用通用方式：取 r.obb.boxes (若存在)，否则退回 r.obb
        try:
            boxes = r.obb.boxes
        except Exception:
            boxes = r.obb

        if len(boxes) == 0:
            return None

        # 取分数最高的
        try:
            scores = boxes.conf.cpu().numpy()
            idx = int(np.argmax(scores))
            xywhr = boxes.xywhr[idx].cpu().numpy()  # (cx, cy, w, h, rot[rad] or deg??)
            angle = float(xywhr[4])
            # 部分版本 angle 可能是度，这里做下判断（经验：|angle|>pi 则当作度）
            if abs(angle) > math.pi * 1.5:
                angle = math.radians(angle)
            cx = float(xywhr[0]) / float(self._w)
            cy = float(xywhr[1]) / float(self._h)
            theta = _rad_wrap(angle)
            return cx, cy, theta
        except Exception:
            # 兼容老字段
            try:
                # 如果只有 poly 顶点：可用 cv2.minAreaRect 拟合
                polys = boxes.xyxyxyxy[idx].cpu().numpy().reshape(-1, 2).astype(np.float32)
                rect = cv2.minAreaRect(polys)
                (cx, cy), (w, h), ang_deg = rect
                # OpenCV 的角度定义和我们期望的有差异，这里做个换算（转弧度 + wrap）
                theta = _rad_wrap(math.radians(ang_deg))
                return float(cx) / self._w, float(cy) / self._h, theta
            except Exception:
                return None

    # --------- 内部：软上限 ----------
    def _soft_angle_limit(self, vtheta_abs: float, dt: float) -> float:
        # 软上限 = k_soft * |vθ| * dt + base_margin
        return self.params.k_soft * vtheta_abs * dt + self._base_margin

    # --------- 工具 ----------
    @property
    def state_dim(self) -> int:
        return 7
