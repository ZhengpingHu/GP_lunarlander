#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# [FIX] 导入 threading 用于并发处理客户端
from threading import Thread
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
# 7D 状态估计器
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
                 smooth_alpha: float = 0.20,
                 gate_hard_deg: float = 20.0,
                 fps_hint: float = 30.0,
                 # [FIX] 允许传入已加载的模型以实现克隆
                 _obb_model: Optional[YOLO] = None,
                 _pose_model: Optional[YOLO] = None):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # [FIX] 优化模型加载逻辑，支持从外部传入已加载的模型
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

        # 保存所有配置参数，以便克隆
        self._init_args = locals()

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
        
        # 每个实例维护自己的状态
        self.state: Optional[np.ndarray] = None

    def clone(self) -> 'YoloStateEstimator7D':
        """
        创建一个新的估计器实例，但共享已加载的YOLO模型。
        这样可以为每个客户端会话提供独立的状态，同时避免重复加载模型的开销。
        """
        args = self._init_args.copy()
        args.pop('self')
        # 传入已加载的模型对象，而不是路径
        args['_obb_model'] = self.obb
        args['_pose_model'] = self.pose
        args['obb_model_path'] = None
        args['pose_model_path'] = None
        return YoloStateEstimator7D(**args)

    def begin_episode(self):
        """重置当前实例的状态"""
        self.state = None

    @torch.no_grad()
    def _infer_obb(self, frame_bgr: np.ndarray) -> Optional[Tuple[float, float, float]]:
        res = self.obb.predict(frame_bgr, **self.pred_kwargs_obb)
        if not res or len(res) == 0: return None
        
        r = res[0]
        boxes = getattr(r, "obb", None)
        if boxes is None or len(boxes) == 0: return None

        scores = boxes.conf.cpu().numpy()
        idx = int(np.argmax(scores))
        b = boxes.xywhr.cpu().numpy()[idx]

        H, W = frame_bgr.shape[:2]
        cx_norm = float(b[0] / W)  # 归一化坐标
        cy_norm = float(b[1] / H)
        theta = wrap_pi(float(b[4]))
        return (cx_norm, cy_norm, theta)

    @torch.no_grad()
    def _infer_pose_theta(self, frame_bgr: np.ndarray) -> Optional[float]:
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
        # 硬门限：如果角度差异太大，则信任 OBB 的结果
        if abs(angle_diff(th_obb, th_pose)) > self.gate_hard:
            return th_obb
        return th_pose

    # [FIX] 接收 steps 参数以匹配客户端调用
    def update_full(self, frame_bgr: np.ndarray, steps: int) -> Optional[np.ndarray]:
        obb_res = self._infer_obb(frame_bgr)
        if obb_res is None:
            self.state = None # 感知失败，清空状态
            return None
            
        cx, cy, th_obb = obb_res
        th_pose = self._infer_pose_theta(frame_bgr)
        th = self._fuse_theta(th_obb, th_pose)

        if self.state is None: # 第一帧
            vx, vy, vth = 0.0, 0.0, 0.0
        else: # 后续帧，计算速度
            cx0, cy0, th0 = self.state[:3]
            vx = (cx - cx0) / self.dt
            vy = (cy - cy0) / self.dt
            vth = angle_diff(th, th0) / self.dt
            
        speed = math.sqrt(vx**2 + vy**2)
        
        self.state = np.array([cx, cy, th, vx, vy, vth, speed], dtype=np.float32)
        return self.state.copy()

    def predict_only(self) -> Optional[np.ndarray]:
        if self.state is None: return None
        
        cx, cy, th, vx, vy, vth, speed = self.state
        # 简单线性外推
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

    # [FIX] 为每个客户端连接创建一个独立的处理器
    def _handle_client(self, conn):
        print("[RPC-SRV] Client connected.")
        # 为此会话克隆一个独立的估计器实例
        session_est = self.master_estimator.clone()
        
        try:
            while True:
                try:
                    msg = conn.recv()
                except EOFError:
                    # 客户端正常关闭连接
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
                        # [FIX] 正确解包 (frame, steps) 元组
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
                    conn.send((False, str(e))) # 将错误信息发回客户端
                    break # 发生严重错误，断开连接

        finally:
            conn.close()
            print("[RPC-SRV] Client disconnected.")

    # [FIX] 正确的服务器主循环
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
                # 为每个连接启动一个新线程，使其可以并发处理
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
    p.add_argument("--fps-hint", type=float, default=30.0, help="Hint for physics timestep calculation.")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=6001)
    p.add_argument("--authkey", type=str, default="yolo-rpc", help="Authentication key for RPC connection.")
    args = p.parse_args()

    # 1. 创建一个“母版”估计器实例，它会加载模型到GPU
    master_est = YoloStateEstimator7D(
        obb_model_path=args.obb_model,
        pose_model_path=args.pose_model,
        device=args.device,
        imgsz_obb=args.imgsz_obb,
        imgsz_pose=args.imgsz_pose,
        conf_obb=args.conf_obb,
        conf_pose=args.conf_pose,
        gate_hard_deg=args.gate_hard_deg,
        fps_hint=args.fps_hint,
    )
    print("[RPC-SRV] Models loaded successfully.")

    # 2. 将母版实例传给服务器。服务器会为每个客户端“克隆”它
    srv = InferenceServer(master_est, host=args.host, port=args.port, authkey=args.authkey.encode("utf-8"))
    srv.serve_forever()

if __name__ == "__main__":
    main()