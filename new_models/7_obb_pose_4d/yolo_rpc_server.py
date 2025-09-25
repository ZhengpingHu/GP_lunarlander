#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import argparse
import socket
import time
from multiprocessing.connection import Listener
from typing import Optional, Tuple, List

import numpy as np
import torch

# OpenMP/BLAS 限制线程，降低冲突
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# 如仍遇到 OMP #15，可再在外部设置：
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(
        f"Import ultralytics failed: {e}\n"
        "Please `pip install ultralytics` and ensure OBB/Pose weights are compatible."
    )

# ------------------------------
# 工具：角度 wrap 与差分
# ------------------------------
def wrap_pi(angle: float) -> float:
    """wrap angle to [-pi, pi]"""
    a = (angle + math.pi) % (2 * math.pi) - math.pi
    return a

def angle_diff(a: float, b: float) -> float:
    """shortest diff a-b in [-pi, pi]"""
    return wrap_pi(a - b)

# ------------------------------
# 4D 状态估计器
# - 目标：输出 [cx_norm, cy_norm, theta_rad, dtheta_rad]
#   * cx_norm, cy_norm：目标 OBB 中心点在 [0,1] 归一化图像坐标
#   * theta_rad：机体姿态角（由 Pose 两关键点估计，或退化用 OBB 角）
#   * dtheta_rad：角速度（相邻观测差分 / dt）
# - 保护：
#   * EMA 平滑
#   * 角度软/硬限制（软限制随当前角速度给出允许的最大变化；硬限制是单侧喷气的极限）
#   * predict_only：基于上次状态 + 常速度外推
# ------------------------------
class YoloStateEstimator4D:
    def __init__(
        self,
        obb_model_path: str,
        pose_model_path: str,
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
        smooth_alpha: float = 0.20,   # EMA α，越大越跟新观测
        gate_soft_k: float = 0.8,     # 角度软限制系数（随 |dθ| 线性放大）
        gate_soft_bias_deg: float = 10.0,  # 角度软限制基础项（度）
        gate_hard_deg: float = 180.0, # 角度硬限制（度）
        fps_hint: float = 30.0,       # 估计 dt=1/fps_hint，用于差分/外推
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 加载模型（常驻 GPU）
        self.obb = YOLO(obb_model_path)
        self.pose = YOLO(pose_model_path)

        self.obb.to(self.device)
        self.pose.to(self.device)

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

        # 平滑与门控
        self.alpha = smooth_alpha
        self.gate_soft_k = gate_soft_k
        self.gate_soft_bias = math.radians(gate_soft_bias_deg)
        self.gate_hard = math.radians(gate_hard_deg)

        # 时间步
        self.dt = 1.0 / max(1.0, fps_hint)

        # 观测缓存
        self.z_prev: Optional[np.ndarray] = None   # 上次“可信”的观测 [cx, cy, theta]
        self.state: Optional[np.ndarray] = None    # 当前 state 4D
        self.t_last: Optional[float] = None

    def begin_episode(self):
        """重置 episode 内部状态"""
        self.z_prev = None
        self.state = None
        self.t_last = None

    # ---------- 内部：一次 OBB 推理 ----------
    @torch.no_grad()
    def _infer_obb(self, frame_bgr: np.ndarray) -> Optional[Tuple[float, float, float]]:
        # 返回 (cx_norm, cy_norm, obb_theta)；若失败返回 None
        # Ultralytics 接受 BGR/ndarray
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

        # 取分数最高的一个
        boxes = r.obb  # xywht 形式（x,y 是中心，w,h, angle）
        scores = r.boxes.conf.cpu().numpy() if hasattr(r, "boxes") and r.boxes is not None else np.ones((len(boxes),), dtype=np.float32)
        idx = int(np.argmax(scores))
        b = boxes.xywhn.cpu().numpy()[idx] if hasattr(boxes, "xywhn") else None
        # 如果没有归一化，就退化使用 xywh
        if b is None:
            b2 = boxes.xywh.cpu().numpy()[idx]
            # 需要根据原图尺寸归一化
            H, W = frame_bgr.shape[:2]
            cx = float(b2[0] / W)
            cy = float(b2[1] / H)
            theta = float(boxes.theta.cpu().numpy()[idx]) if hasattr(boxes, "theta") else 0.0
        else:
            cx = float(b[0])
            cy = float(b[1])
            theta = float(boxes.theta.cpu().numpy()[idx]) if hasattr(boxes, "theta") else 0.0

        # OBB 的 theta 定义可能是 [-pi/2, pi/2]，统一 wrap
        theta = wrap_pi(theta)
        return (cx, cy, theta)

    # ---------- 内部：一次 Pose 推理 ----------
    @torch.no_grad()
    def _infer_pose_theta(self, frame_bgr: np.ndarray) -> Optional[float]:
        # 用两个关键点估计机体朝向（例如主体中心->喷口/腿的连线）
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

        # 取分数最高的人/目标
        kp = r.keypoints  # shape: (n, num_kpts, 2 or 3)
        scores = r.boxes.conf.cpu().numpy() if hasattr(r, "boxes") and r.boxes is not None else np.ones((len(kp),), dtype=np.float32)
        idx = int(np.argmax(scores))
        k = kp.xyn[idx].cpu().numpy() if hasattr(kp, "xyn") else kp.xy[idx].cpu().numpy()

        # 这里假设第 0/1 号关键点能决定机体朝向（按你的数据集定义修改）
        if k.shape[0] < 2:
            return None
        p0 = k[0]
        p1 = k[1]
        dx, dy = (p1[0] - p0[0]), (p1[1] - p0[1])
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        theta = math.atan2(dy, dx)
        # wrap
        theta = wrap_pi(theta)
        return theta

    # ---------- 内部：融合 OBB 与 Pose 角 ----------
    def _fuse_theta(self, theta_obb: Optional[float], theta_pose: Optional[float]) -> Optional[float]:
        if theta_pose is not None:
            return theta_pose
        return theta_obb

    # ---------- 内部：角度门控（软/硬） + EMA 平滑 ----------
    def _gate_and_smooth(self, z_new: np.ndarray) -> np.ndarray:
        # z_new: [cx, cy, theta]
        if self.z_prev is None:
            self.z_prev = z_new.copy()
            return z_new.copy()

        cx0, cy0, th0 = self.z_prev
        cx1, cy1, th1 = z_new

        # 角度差
        dth = angle_diff(th1, th0)

        # 软限制：|Δθ| <= bias + k * |dθ_prev|
        dtheta_prev = 0.0
        if self.state is not None:
            dtheta_prev = float(self.state[3])  # 上次角速度
        soft_limit = self.gate_soft_bias + self.gate_soft_k * abs(dtheta_prev)
        soft_limit = min(soft_limit, self.gate_hard)

        # 硬限制：|Δθ| <= gate_hard
        hard_limit = self.gate_hard

        # 先应用硬限制，再应用软限制
        limited = max(-hard_limit, min(hard_limit, dth))
        limited = max(-soft_limit, min(soft_limit, limited))

        # 生成门控后的角
        th_g = wrap_pi(th0 + limited)

        # 位置直接 EMA，角度也 EMA（但用 th_g）
        cx_s = self.alpha * cx1 + (1.0 - self.alpha) * cx0
        cy_s = self.alpha * cy1 + (1.0 - self.alpha) * cy0

        # 角度 EMA 要用短差
        th_s = wrap_pi(th0 + self.alpha * angle_diff(th_g, th0))

        out = np.array([cx_s, cy_s, th_s], dtype=np.float32)
        self.z_prev = out.copy()
        return out

    # ---------- PRC API：一次完整观测（刷新速度） ----------
    def update_full(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        # 1) OBB 检测中心 + 粗角
        obb = self._infer_obb(frame_bgr)
        if obb is None:
            return None
        cx, cy, th_obb = obb

        # 2) Pose 估计角（若可用）
        th_pose = self._infer_pose_theta(frame_bgr)

        # 3) 融合角
        th = self._fuse_theta(th_obb, th_pose)
        if th is None:
            return None

        # 4) 门控 + 平滑（输出平滑的位置与角）
        z = np.array([cx, cy, th], dtype=np.float32)
        z_bar = self._gate_and_smooth(z)

        # 5) 速度估计（角速度）；位置速度你当前 4D 不用，这里只给 dθ
        if self.state is None:
            dtheta = 0.0
        else:
            th_prev = float(self.state[2])
            dtheta = angle_diff(float(z_bar[2]), th_prev) / self.dt

        self.state = np.array([z_bar[0], z_bar[1], z_bar[2], dtheta], dtype=np.float32)
        self.t_last = time.time()
        return self.state.copy()

    # ---------- RPC API：预测外推（常速度） ----------
    def predict_only(self) -> Optional[np.ndarray]:
        if self.state is None:
            return None
        # 常速度外推角度
        cx, cy, th, dth = [float(x) for x in self.state]
        th2 = wrap_pi(th + dth * self.dt)
        # 保持位置不动（也可以轻微衰减）
        self.state = np.array([cx, cy, th2, dth], dtype=np.float32)
        return self.state.copy()

# ------------------------------
# 简单的顺序式 RPC 服务器（单进程、单 GPU）
# ------------------------------
class InferenceServer:
    def __init__(
        self,
        obb_model: str,
        pose_model: str,
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
        smooth_alpha: float = 0.20,
        gate_soft_k: float = 0.8,
        gate_soft_bias_deg: float = 10.0,
        gate_hard_deg: float = 180.0,
        fps_hint: float = 30.0,
        host: str = "127.0.0.1",
        port: int = 6001,
        authkey: bytes = b"yolo-rpc",
        backlog: int = 2,
    ):
        print(f"[RPC-SRV] Loading models on device '{device}' ...")
        self.est = YoloStateEstimator4D(
            obb_model_path=obb_model,
            pose_model_path=pose_model,
            device=device,
            imgsz_obb=imgsz_obb,
            imgsz_pose=imgsz_pose,
            conf_obb=conf_obb,
            conf_pose=conf_pose,
            iou_obb=iou_obb,
            iou_pose=iou_pose,
            max_det=max_det,
            agnostic_nms=agnostic_nms,
            pad=pad,
            smooth_alpha=smooth_alpha,
            gate_soft_k=gate_soft_k,
            gate_soft_bias_deg=gate_soft_bias_deg,
            gate_hard_deg=gate_hard_deg,
            fps_hint=fps_hint,
        )
        self.est.begin_episode()
        print("[RPC-SRV] Models loaded.")

        self.address = (host, port)
        self.authkey = authkey
        self.backlog = backlog

    def serve_forever(self):
        print(f"[RPC-SRV] Listening on {self.address} ...")
        # 注意：multiprocessing Listener 在 Windows 下 backlog 不可直接调，
        # 这里采用“短连接顺序处理”策略：一次只处理一个 client，处理完立即关闭。
        while True:
            try:
                listener = Listener(self.address, authkey=self.authkey)
            except OSError as e:
                print(f"[RPC-SRV] Listener bind failed: {e}. Maybe port busy. Retry in 2s.")
                time.sleep(2.0)
                continue

            try:
                conn = listener.accept()
                print("[RPC-SRV] client connected.")
                try:
                    while True:
                        try:
                            msg = conn.recv()
                        except EOFError:
                            break
                        if not isinstance(msg, tuple) or len(msg) != 2:
                            conn.send(False)
                            continue
                        cmd, payload = msg

                        if cmd == "ping":
                            conn.send(True)

                        elif cmd == "reset":
                            self.est.begin_episode()
                            conn.send(True)

                        elif cmd == "update_full":
                            frame_bgr = payload
                            try:
                                out = self.est.update_full(frame_bgr)
                                if out is None:
                                    conn.send((False, None))
                                else:
                                    conn.send((True, out))
                            except Exception as e:
                                conn.send((False, None))

                        elif cmd == "predict_only":
                            out = self.est.predict_only()
                            if out is None:
                                conn.send((False, None))
                            else:
                                conn.send((True, out))

                        else:
                            conn.send(False)
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    print("[RPC-SRV] client disconnected.")
            finally:
                try:
                    listener.close()
                except Exception:
                    pass

# ------------------------------
# CLI
# ------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obb-model", type=str, required=True, help="YOLO OBB weight (.pt)")
    p.add_argument("--pose-model", type=str, required=True, help="YOLO Pose weight (.pt)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--imgsz-obb", type=int, default=640)
    p.add_argument("--imgsz-pose", type=int, default=384)
    p.add_argument("--conf-obb", type=float, default=0.25)
    p.add_argument("--conf-pose", type=float, default=0.20)
    p.add_argument("--iou-obb", type=float, default=0.7)
    p.add_argument("--iou-pose", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=10)
    p.add_argument("--agnostic-nms", action="store_true", default=True)
    p.add_argument("--pad", type=float, default=0.25)
    p.add_argument("--smooth-alpha", type=float, default=0.20)
    p.add_argument("--gate-soft-k", type=float, default=0.8)
    p.add_argument("--gate-soft-bias-deg", type=float, default=10.0)
    p.add_argument("--gate-hard-deg", type=float, default=180.0)
    p.add_argument("--fps-hint", type=float, default=30.0)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=6001)
    p.add_argument("--authkey", type=str, default="yolo-rpc")
    args = p.parse_args()

    srv = InferenceServer(
        obb_model=args.obb_model,
        pose_model=args.pose_model,
        device=args.device,
        imgsz_obb=args.imgsz_obb,
        imgsz_pose=args.imgsz_pose,
        conf_obb=args.conf_obb,
        conf_pose=args.conf_pose,
        iou_obb=args.iou_obb,
        iou_pose=args.iou_pose,
        max_det=args.max_det,
        agnostic_nms=args.agnostic_nms,
        pad=args.pad,
        smooth_alpha=args.smooth_alpha,
        gate_soft_k=args.gate_soft_k,
        gate_soft_bias_deg=args.gate_soft_bias_deg,
        gate_hard_deg=args.gate_hard_deg,
        fps_hint=args.fps_hint,
        host=args.host,
        port=args.port,
        authkey=args.authkey.encode("utf-8"),
    )
    srv.serve_forever()

if __name__ == "__main__":
    main()
