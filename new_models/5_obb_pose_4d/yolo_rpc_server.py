#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 推理 RPC 服务端（Windows/Ubuntu 通用）
- 只在这里加载 YOLO 模型（GPU/CPU 任选），所有训练进程通过 RPC 共享
- 支持命令：'infer'（推理）、'predict'（本地外推）、'reset'（清历史）、'ping'
- 已加：推理串行锁（默认开启，最稳），异常打印，简易健康日志
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 消除 OMP #15 警告
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")      # 默认用 0 号卡；如用 CPU 改为空字符串
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
import argparse
import traceback
import threading
from multiprocessing.connection import Listener
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2

# 你的 4D 状态估计器（与之前一致）
from yolo_state_4d import YoloStateEstimator4D


_INFER_LOCK = threading.Lock()  # → 串行化推理更稳；稳定后可去掉
PRINT_PREFIX = "[RPC-SRV]"


def _send(conn, obj: Dict[str, Any]) -> None:
    """用 connection 的内置序列化发送（比手写长度前缀简单稳）"""
    conn.send(obj)


def _recv(conn) -> Optional[Dict[str, Any]]:
    try:
        obj = conn.recv()
        if isinstance(obj, dict):
            return obj
        return None
    except EOFError:
        return None
    except Exception:
        return None


class InferenceServer:
    def __init__(
        self,
        address: Tuple[str, int],
        authkey: bytes,
        obb_model: str,
        pose_model: str,
        device: str = "cuda:0",
        imgsz_obb: int = 640,
        imgsz_pose: int = 384,
        conf_obb: float = 0.25,
        conf_pose: float = 0.20,
        pad: float = 0.25,
        gate_deg: float = 120.0,
        smooth_alpha: float = 0.2,
        yolo_freq: int = 8,  # 仅用于日志；实际由客户端控制
    ) -> None:
        self.address = address
        self.authkey = authkey
        self.device = device

        print(PRINT_PREFIX, f"Loading models on device '{device}' ...")
        self.est = YoloStateEstimator4D(
            obb_model_path=obb_model,
            pose_model_path=pose_model,
            device=device,
            imgsz_obb=imgsz_obb,
            imgsz_pose=imgsz_pose,
            conf_obb=conf_obb,
            conf_pose=conf_pose,
            base_pad=pad,
            gate_deg=gate_deg,
            smooth_alpha=smooth_alpha,
        )
        self.est.begin_episode()
        self.yolo_freq = yolo_freq
        print(PRINT_PREFIX, "Models loaded.")

    def _handle_conn(self, conn) -> None:
        """每个连接一个线程，处理循环请求"""
        try:
            while True:
                req = _recv(conn)
                if req is None:
                    print(PRINT_PREFIX, "client disconnected.")
                    break

                cmd = req.get("cmd", None)

                if cmd == "ping":
                    _send(conn, {"ok": True, "pong": True})
                    continue

                if cmd == "reset":
                    self.est.begin_episode()
                    _send(conn, {"ok": True})
                    continue

                if cmd == "infer":
                    frame = req.get("frame", None)
                    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
                        _send(conn, {"ok": False, "msg": "bad-frame"})
                        continue
                    try:
                        with _INFER_LOCK:
                            z = self.est.update_full(frame)
                    except Exception as e:
                        print(PRINT_PREFIX, "infer error:", repr(e))
                        traceback.print_exc()
                        _send(conn, {"ok": False, "msg": f"infer-exc: {repr(e)}"})
                        continue
                    if z is None:
                        _send(conn, {"ok": False, "msg": "no-detection"})
                    else:
                        _send(conn, {"ok": True, "z": z})
                    continue

                if cmd == "predict":
                    try:
                        z = self.est.predict_only()
                    except Exception as e:
                        print(PRINT_PREFIX, "predict error:", repr(e))
                        traceback.print_exc()
                        _send(conn, {"ok": False, "msg": f"predict-exc: {repr(e)}"})
                        continue
                    if z is None:
                        _send(conn, {"ok": False, "msg": "no-state"})
                    else:
                        _send(conn, {"ok": True, "z": z})
                    continue

                # 未知命令
                _send(conn, {"ok": False, "msg": f"unknown-cmd:{cmd}"})

        except Exception as e:
            print(PRINT_PREFIX, "handler crash:", repr(e))
            traceback.print_exc()
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def serve_forever(self) -> None:
        print(PRINT_PREFIX, f"Listening on {self.address} ...")
        listener = Listener(self.address, authkey=self.authkey)
        while True:
            conn = listener.accept()
            print(PRINT_PREFIX, "client connected.")
            t = threading.Thread(target=self._handle_conn, args=(conn,), daemon=True)
            t.start()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6000)
    p.add_argument("--auth", default="yolo-rpc")  # 简单共享密钥
    p.add_argument("--obb-model", required=True)
    p.add_argument("--pose-model", required=True)
    p.add_argument("--device", default="cuda:0", help="cuda:0 / cpu")
    p.add_argument("--imgsz-obb", type=int, default=640)
    p.add_argument("--imgsz-pose", type=int, default=384)
    p.add_argument("--conf-obb", type=float, default=0.25)
    p.add_argument("--conf-pose", type=float, default=0.20)
    p.add_argument("--pad", type=float, default=0.25)
    p.add_argument("--gate-deg", type=float, default=120.0)
    p.add_argument("--smooth-alpha", type=float, default=0.2)
    p.add_argument("--yolo-freq", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    srv = InferenceServer(
        address=(args.host, args.port),
        authkey=args.auth.encode("utf-8"),
        obb_model=args.obb_model,
        pose_model=args.pose_model,
        device=args.device,
        imgsz_obb=args.imgsz_obb,
        imgsz_pose=args.imgsz_pose,
        conf_obb=args.conf_obb,
        conf_pose=args.conf_pose,
        pad=args.pad,
        gate_deg=args.gate_deg,
        smooth_alpha=args.smooth_alpha,
        yolo_freq=args.yolo_freq,
    )
    srv.serve_forever()
