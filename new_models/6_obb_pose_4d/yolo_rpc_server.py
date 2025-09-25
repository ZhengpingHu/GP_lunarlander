#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
import argparse
import traceback
from multiprocessing.connection import Listener
from threading import Thread

AUTHKEY = b"yolo-rpc"  # ← 客户端必须与此完全一致

# 让 stdout 及时刷新（Windows 下方便看日志）
print = lambda *a, **k: (__import__("builtins").print(*a, **{**k, "flush": True}))

# --------- 导入你的状态估计器 ----------
try:
    from yolo_state_4d import YoloStateEstimator4D  # 需要 begin_episode/update_full/predict_only
except Exception as e:
    print("[RPC-SRV][FATAL] 无法 import yolo_state_4d.YoloStateEstimator4D：", e)
    traceback.print_exc()
    sys.exit(1)


class InferenceServer:
    def __init__(
        self,
        obb_model_path: str,
        pose_model_path: str,
        device: str = "cuda:0",
        imgsz_obb: int = 640,
        imgsz_pose: int = 384,
        conf_obb: float = 0.25,
        conf_pose: float = 0.20,
        base_pad: float = 0.25,
        gate_deg: float = 120.0,
        smooth_alpha: float = 0.2,
        host: str = "127.0.0.1",
        port: int = 6000,
    ):
        self.address = (host, port)
        self.authkey = AUTHKEY

        print(f"[RPC-SRV] Loading models on device '{device}' ...")
        self.est = YoloStateEstimator4D(
            obb_model_path=obb_model_path,
            pose_model_path=pose_model_path,
            device=device,
            conf_obb=conf_obb,
            conf_pose=conf_pose,
            imgsz_obb=imgsz_obb,
            imgsz_pose=imgsz_pose,
            base_pad=base_pad,
            gate_deg=gate_deg,
            smooth_alpha=smooth_alpha,
        )
        self.est.begin_episode()
        print("[RPC-SRV] Models loaded.")

    # 处理单个客户端
    def _handle_client(self, conn):
        print("[RPC-SRV] client connected.")
        try:
            while True:
                try:
                    req = conn.recv()
                except EOFError:
                    break
                except Exception as e:
                    print(f"[RPC-SRV] recv error: {e}")
                    break

                if not isinstance(req, tuple) or len(req) != 2:
                    conn.send((False, "bad request"))
                    continue

                method, payload = req

                if method == "ping":
                    conn.send(("pong", time.time()))
                elif method == "reset":
                    self.est.begin_episode()
                    conn.send((True, None))
                elif method == "update_full":
                    frame_bgr = payload
                    try:
                        z = self.est.update_full(frame_bgr)
                        conn.send((True, z if z is not None else None))
                    except Exception as e:
                        print("[RPC-SRV] update_full error:", e)
                        traceback.print_exc()
                        conn.send((False, None))
                elif method == "predict_only":
                    try:
                        z = self.est.predict_only()
                        conn.send((True, z if z is not None else None))
                    except Exception as e:
                        print("[RPC-SRV] predict_only error:", e)
                        traceback.print_exc()
                        conn.send((False, None))
                else:
                    conn.send((False, f"unknown method: {method}"))
        finally:
            try:
                conn.close()
            except Exception:
                pass
            print("[RPC-SRV] client disconnected.")

    def serve_forever(self):
        # 可能端口被占用时更友好
        listener = None
        try:
            listener = Listener(self.address, authkey=self.authkey)
        except OSError as e:
            print(f"[RPC-SRV][FATAL] 监听端口失败 {self.address}: {e}")
            print("  - 请确认端口未被占用（换个 --port 试试）")
            sys.exit(1)

        print(f"[RPC-SRV] Listening on {self.address} ...")

        while True:
            try:
                conn = listener.accept()
            except Exception as e:
                print("[RPC-SRV] listener.accept error:", e)
                time.sleep(0.2)
                continue

            Thread(target=self._handle_client, args=(conn,), daemon=True).start()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obb-model", type=str, required=True)
    ap.add_argument("--pose-model", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--imgsz-obb", type=int, default=640)
    ap.add_argument("--imgsz-pose", type=int, default=384)
    ap.add_argument("--conf-obb", type=float, default=0.25)
    ap.add_argument("--conf-pose", type=float, default=0.20)
    ap.add_argument("--pad", type=float, default=0.25)
    ap.add_argument("--gate-deg", type=float, default=120.0)
    ap.add_argument("--smooth-alpha", type=float, default=0.2)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6000)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    srv = InferenceServer(
        obb_model_path=args.obb_model,
        pose_model_path=args.pose_model,
        device=args.device,
        imgsz_obb=args.imgsz_obb,
        imgsz_pose=args.imgsz_pose,
        conf_obb=args.conf_obb,
        conf_pose=args.conf_pose,
        base_pad=args.pad,
        gate_deg=args.gate_deg,
        smooth_alpha=args.smooth_alpha,
        host=args.host,
        port=args.port,
    )
    srv.serve_forever()
