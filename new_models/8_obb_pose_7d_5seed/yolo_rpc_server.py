#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# yolo_rpc_server.py
import os
from threading import Thread, Lock
import math
import argparse
import traceback
from multiprocessing.connection import Listener
from typing import Optional, Tuple

import numpy as np
import torch

# 限制线程冲突
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(f"Import ultralytics failed: {e}")

# 让 stdout 及时刷新
print = lambda *a, **k: (__import__("builtins").print(*a, **{**k, "flush": True}))


# -------------------
# 工具函数
# -------------------
def wrap_pi(angle: float) -> float:
    """将角度约束在 (-pi, pi] 范围内"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def angle_diff(a: float, b: float) -> float:
    """计算两个角度之间的最短差值"""
    return wrap_pi(a - b)

# -------------------
# 7D 状态估计器（线程安全 predict + 单实例 YOLO + EMA平滑）
# -------------------
class YoloStateEstimator7D:
    def __init__(self,
                 obb_model_path: Optional[str] = None,
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
                 smooth_alpha: float = 0.6,  # [MODIFIED] 平滑系数 (alpha越大, 新观测权重越高)
                 gate_hard_deg: float = 20.0,
                 fps_hint: float = 30.0,
                 _obb_model: Optional[YOLO] = None,
                 _pose_model: Optional[YOLO] = None,
                 _skip_warmup: bool = False
                 ):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 加载/复用模型
        if _obb_model:
            self.obb = _obb_model
        elif obb_model_path:
            self.obb = YOLO(obb_model_path).to(self.device)
        else:
            raise ValueError("Either obb_model_path or _obb_model must be provided.")

        if _pose_model:
            self.pose = _pose_model
        elif pose_model_path:
            self.pose = YOLO(pose_model_path).to(self.device)
        else:
            raise ValueError("Either pose_model_path or _pose_model must be provided.")

        # 预测参数
        self.pred_kwargs_obb = dict(
            imgsz=imgsz_obb, conf=conf_obb, iou=iou_obb, max_det=max_det,
            agnostic_nms=agnostic_nms, device=self.device, verbose=False
        )
        self.pred_kwargs_pose = dict(
            imgsz=imgsz_pose, conf=conf_pose, iou=iou_pose, max_det=max_det,
            agnostic_nms=agnostic_nms, device=self.device, verbose=False
        )
        
        self.gate_hard = math.radians(gate_hard_deg)
        self.dt = 1.0 / max(1.0, fps_hint)
        self.smooth_alpha = smooth_alpha
        
        # [MODIFIED] self.state 存储平滑后的值, raw_pos_theta 存储上一帧原始观测值
        self.state: Optional[np.ndarray] = None
        self.raw_pos_theta: Optional[np.ndarray] = None

        # 线程锁
        self._lock_obb = Lock()
        self._lock_pose = Lock()

        # [MODIFIED] 保存初始化参数, 已包含 smooth_alpha
        self._init_args = dict(
            obb_model_path=obb_model_path, pose_model_path=pose_model_path,
            device=device, imgsz_obb=imgsz_obb, imgsz_pose=imgsz_pose,
            conf_obb=conf_obb, conf_pose=conf_pose, iou_obb=iou_obb, iou_pose=iou_pose,
            max_det=max_det, agnostic_nms=agnostic_nms, smooth_alpha=smooth_alpha,
            gate_hard_deg=gate_hard_deg, fps_hint=fps_hint
        )

        # Warmup
        if not _skip_warmup:
            try:
                dummy_obb = np.zeros((int(imgsz_obb), int(imgsz_obb), 3), dtype=np.uint8)
                dummy_pose = np.zeros((int(imgsz_pose), int(imgsz_pose), 3), dtype=np.uint8)
                with self._lock_obb:
                    _ = self.obb.predict(dummy_obb, **self.pred_kwargs_obb)
                with self._lock_pose:
                    _ = self.pose.predict(dummy_pose, **self.pred_kwargs_pose)
            except Exception as e:
                print(f"[WARN] Warmup failed (safe to continue): {e}")

    def clone(self) -> 'YoloStateEstimator7D':
        args = dict(self._init_args)
        args['_obb_model'] = self.obb
        args['_pose_model'] = self.pose
        args['_skip_warmup'] = True
        new_est = YoloStateEstimator7D(**args)
        new_est._lock_obb = self._lock_obb
        new_est._lock_pose = self._lock_pose
        return new_est

    def begin_episode(self):
        """重置当前实例的状态"""
        self.state = None
        self.raw_pos_theta = None # [MODIFIED] 也要重置原始观测

    @torch.no_grad()
    def _infer_obb(self, frame_bgr: np.ndarray) -> Optional[Tuple[float, float, float]]:
        with self._lock_obb:
            res = self.obb.predict(frame_bgr, **self.pred_kwargs_obb)
        if not res or len(res) == 0: return None
        r = res[0]
        boxes = getattr(r, "obb", None)
        if boxes is None or len(boxes) == 0: return None
        scores = boxes.conf.cpu().numpy()
        idx = int(np.argmax(scores))
        b = boxes.xywhr.cpu().numpy()[idx]
        H, W = frame_bgr.shape[:2]
        cx_norm = float(b[0] / W)
        cy_norm = float(b[1] / H)
        theta = wrap_pi(float(b[4]))
        return (cx_norm, cy_norm, theta)

    @torch.no_grad()
    def _infer_pose_theta(self, frame_bgr: np.ndarray) -> Optional[float]:
        with self._lock_pose:
            res = self.pose.predict(frame_bgr, **self.pred_kwargs_pose)
        if not res or len(res) == 0: return None
        r = res[0]
        if r.keypoints is None or len(r.keypoints) == 0 or r.boxes is None or len(r.boxes) == 0:
            return None
        idx = int(np.argmax(r.boxes.conf.cpu().numpy()))
        k = r.keypoints.xyn[idx].cpu().numpy()
        if k.shape[0] < 2: return None
        p0, p1 = k[0], k[1]
        dx, dy = (p1[0] - p0[0]), (p1[1] - p0[1])
        return wrap_pi(math.atan2(dy, dx)) if abs(dx) > 1e-6 or abs(dy) > 1e-6 else None

    def _fuse_theta(self, th_obb: float, th_pose: Optional[float]) -> float:
        if th_pose is None:
            return th_obb
        if abs(angle_diff(th_obb, th_pose)) > self.gate_hard:
            return th_obb
        return th_pose

    def update_full(self, frame_bgr: np.ndarray, steps: int) -> Optional[np.ndarray]:
        # [MODIFIED] 核心修改逻辑: 实现了EMA平滑并修复了预估链条
        obb_res = self._infer_obb(frame_bgr)
        if obb_res is None:
            # 正确！不清除 self.state，让 predict_only 可以工作
            return None
            
        cx, cy, th_obb = obb_res
        th_pose = self._infer_pose_theta(frame_bgr)
        th = self._fuse_theta(th_obb, th_pose)

        # 当前帧的原始(未平滑)观测
        current_raw_pos_theta = np.array([cx, cy, th])

        if self.state is None: # 第一帧
            self.raw_pos_theta = current_raw_pos_theta
            smooth_pos_theta = current_raw_pos_theta
            vx, vy, vth = 0.0, 0.0, 0.0
        else: # 后续帧
            # 1. 平滑位置和角度
            prev_smooth_pos_theta = self.state[:3]
            smooth_pos_theta = self.smooth_alpha * current_raw_pos_theta + (1 - self.smooth_alpha) * prev_smooth_pos_theta
            smooth_pos_theta[2] = wrap_pi(smooth_pos_theta[2]) # 角度需要wrap

            # 2. 计算瞬时速度 (用最新的平滑位置 和 上一帧的原始位置)
            prev_raw_pos_theta = self.raw_pos_theta
            vx_inst = (smooth_pos_theta[0] - prev_raw_pos_theta[0]) / self.dt
            vy_inst = (smooth_pos_theta[1] - prev_raw_pos_theta[1]) / self.dt
            vth_inst = angle_diff(smooth_pos_theta[2], prev_raw_pos_theta[2]) / self.dt
            
            # 3. 平滑速度
            prev_velocities = self.state[3:6]
            current_velocities = np.array([vx_inst, vy_inst, vth_inst])
            smooth_velocities = self.smooth_alpha * current_velocities + (1 - self.smooth_alpha) * prev_velocities
            vx, vy, vth = smooth_velocities
            
            # 4. 更新上一帧的原始观测，为下一次计算做准备
            self.raw_pos_theta = current_raw_pos_theta
            
        speed = math.sqrt(vx**2 + vy**2)
        
        self.state = np.array([
            smooth_pos_theta[0], smooth_pos_theta[1], smooth_pos_theta[2], 
            vx, vy, vth, speed
        ], dtype=np.float32)
        return self.state.copy()

    def predict_only(self) -> Optional[np.ndarray]:
        # (逻辑不变) - 它会自动使用 self.state 中平滑后的速度，因此预测更稳定
        if self.state is None: return None
        
        cx, cy, th, vx, vy, vth, speed = self.state
        cx2 = cx + vx * self.dt
        cy2 = cy + vy * self.dt
        th2 = wrap_pi(th + vth * self.dt)
        
        self.state = np.array([cx2, cy2, th2, vx, vy, vth, speed], dtype=np.float32)
        return self.state.copy()


# -------------------
# RPC Server (Robust Architecture)
# -------------------
class InferenceServer:
    def __init__(self, est: YoloStateEstimator7D, host="127.0.0.1", port=6001, authkey=b"yolo-rpc"):
        self.master_estimator = est
        self.address = (host, port)
        self.authkey = authkey

    def _handle_client(self, conn):
        print("[RPC-SRV] Client connected.")
        session_est = self.master_estimator.clone()
        
        try:
            while True:
                try:
                    msg = conn.recv()
                except EOFError:
                    break
                except Exception as e:
                    print(f"[RPC-SRV] Recv error: {e}")
                    break

                if not isinstance(msg, tuple) or len(msg) != 2:
                    conn.send((False, "bad request"))
                    continue

                cmd, payload = msg
                
                try:
                    if cmd == "ping":
                        conn.send((True, "pong"))
                    elif cmd == "reset":
                        session_est.begin_episode()
                        conn.send((True, None))
                    elif cmd == "update_full":
                        if isinstance(payload, tuple) and len(payload) == 2:
                            frame_bgr, steps = payload
                            out = session_est.update_full(frame_bgr, steps)
                            conn.send((out is not None, out))
                        else:
                            conn.send((False, "invalid payload for update_full"))
                    elif cmd == "predict_only":
                        out = session_est.predict_only()
                        conn.send((out is not None, out))
                    else:
                        conn.send((False, f"unknown command: {cmd}"))
                except Exception as e:
                    print(f"[RPC-SRV] Error processing command '{cmd}': {e}")
                    traceback.print_exc()
                    conn.send((False, str(e)))
                    break
        finally:
            conn.close()
            print("[RPC-SRV] Client disconnected.")

    def serve_forever(self):
        try:
            listener = Listener(self.address, authkey=self.authkey)
        except OSError as e:
            print(f"[FATAL] Failed to listen on {self.address}: {e}")
            print("  - Is the port already in use?")
            return
            
        print(f"[RPC-SRV] Listening on {self.address} ...")
        
        while True:
            try:
                conn = listener.accept()
                client_thread = Thread(target=self._handle_client, args=(conn,), daemon=True)
                client_thread.start()
            except Exception as e:
                print(f"[RPC-SRV] Accept failed: {e}")
                break
        
        listener.close()
        print("[RPC-SRV] Server shutting down.")


# -------------------
# CLI
# -------------------
def main():
    p = argparse.ArgumentParser(description="YOLO OBB+Pose RPC Server for LunarLander")
    p.add_argument("--obb-model", type=str, required=True, help="Path to OBB model")
    p.add_argument("--pose-model", type=str, required=True, help="Path to Pose model")
    p.add_argument("--device", type=str, default="cuda:0", help="Device to run on (e.g., 'cuda:0' or 'cpu')")
    p.add_argument("--imgsz-obb", type=int, default=640)
    p.add_argument("--imgsz-pose", type=int, default=384)
    p.add_argument("--conf-obb", type=float, default=0.25)
    p.add_argument("--conf-pose", type=float, default=0.20)
    p.add_argument("--gate-hard-deg", type=float, default=20.0, help="Hard gate threshold in degrees for fusing OBB and Pose angles.")
    # [NEW] 添加了 smooth-alpha 命令行参数
    p.add_argument("--smooth-alpha", type=float, default=0.6, help="EMA smoothing factor for state estimation.")
    p.add_argument("--fps-hint", type=float, default=30.0, help="Hint for physics timestep calculation.")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=6001)
    p.add_argument("--authkey", type=str, default="yolo-rpc", help="Authentication key for RPC connection.")
    args = p.parse_args()

    # 创建“母版”估计器实例
    master_est = YoloStateEstimator7D(
        obb_model_path=args.obb_model,
        pose_model_path=args.pose_model,
        device=args.device,
        imgsz_obb=args.imgsz_obb,
        imgsz_pose=args.imgsz_pose,
        conf_obb=args.conf_obb,
        conf_pose=args.conf_pose,
        gate_hard_deg=args.gate_hard_deg,
        smooth_alpha=args.smooth_alpha, # [MODIFIED] 确保传入此参数
        fps_hint=args.fps_hint,
    )
    print("[RPC-SRV] Models loaded successfully.")

    # 将母版实例传给服务器
    srv = InferenceServer(master_est, host=args.host, port=args.port, authkey=args.authkey.encode("utf-8"))
    srv.serve_forever()

if __name__ == "__main__":
    main()