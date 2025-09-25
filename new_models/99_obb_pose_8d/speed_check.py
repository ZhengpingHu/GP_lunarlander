#!/usr/bin/env python3
# speed_check.py  (Windows + CUDA, PT only, with angle constraints stats)
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import math
import argparse
from collections import deque

import numpy as np
import cv2
import torch
import gymnasium as gym
from ultralytics import YOLO

from yolo_state_mp_frame_skip_fused import FusedStateEstimator

def torch_sync(device):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

def draw_obb_on(frame, xywhr, color=(0,255,255), thickness=2):
    if xywhr is None:
        return frame
    cx, cy, w, h, th = xywhr
    c = np.array([cx, cy], dtype=np.float32)
    cos_t = math.cos(th); sin_t = math.sin(th)
    dx = w/2.0; dy = h/2.0
    pts = np.array([[-dx,-dy],[dx,-dy],[dx,dy],[-dx,dy]], dtype=np.float32)
    R = np.array([[cos_t, -sin_t],[sin_t, cos_t]], dtype=np.float32)
    pts = (pts @ R.T) + c
    pts = pts.astype(int)
    cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)
    cv2.circle(frame, (int(cx),int(cy)), 3, (0,0,255), -1, cv2.LINE_AA)
    return frame

def main():
    ap = argparse.ArgumentParser(description="YOLO OBB+Pose speed check (PT-only, CUDA/CPU)")
    ap.add_argument("--obb", default="./lander.pt", help="Path to OBB .pt")
    ap.add_argument("--pose", default="./lander-pose.pt", help="Path to Pose .pt")
    ap.add_argument("--imgsz-obb", type=int, default=640, help="OBB imgsz (match training/export)")
    ap.add_argument("--imgsz-pose", type=int, default=384, help="Pose imgsz (match training/export)")
    ap.add_argument("--conf-obb", type=float, default=0.25, help="OBB conf")
    ap.add_argument("--conf-pose", type=float, default=0.20, help="Pose conf")
    ap.add_argument("--pad", type=float, default=0.25, help="crop padding ratio around OBB")
    ap.add_argument("--frames", type=int, default=600, help="Total frames to run")
    ap.add_argument("--freq", type=int, default=10, help="Do full OBB+Pose every N frames (frame-skip)")
    ap.add_argument("--visualize", action="store_true", help="Show cv2 window")
    ap.add_argument("--device", default="auto", choices=["auto","cuda","cpu"], help="Preferred device")
    # 角度约束参数（可外部调）
    ap.add_argument("--hard-deg-s", type=float, default=220.0, help="hard limit of angular velocity (deg/s)")
    ap.add_argument("--soft-floor", type=float, default=30.0, help="soft floor (deg/s)")
    ap.add_argument("--soft-gain", type=float, default=0.50, help="soft gain (deg per pix per s)")
    ap.add_argument("--alpha", type=float, default=0.2, help="angle smoothing alpha (0~1)")
    ap.add_argument("--gate-deg", type=float, default=120.0, help="pose-vs-obb axis gate (deg)")
    args = ap.parse_args()

    # 设备选择仅用于同步
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[警告] 选择了 CUDA 但不可用，退回 CPU")
        device = "cpu"
    elif args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[信息] 使用设备: {device}")

    # 构造状态估计器（内部加载 .pt）
    est = FusedStateEstimator(
        obb_model_path=args.obb,
        pose_model_path=args.pose,
        conf_obb=args.conf_obb,
        conf_pose=args.conf_pose,
        imgsz_obb=args.imgsz_obb,
        imgsz_pose=args.imgsz_pose,
        base_pad=args.pad,
        gate_deg=args.gate_deg,
        smooth_alpha=args.alpha,
        max_spin_hard_deg_s=args.hard_deg_s,
        soft_floor_deg_s=args.soft_floor,
        soft_gain_deg_per_pix_s=args.soft_gain,
        device=args.device,
        frameskip_period=args.freq
    )

    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    obs, _ = env.reset()
    est.begin_episode()

    # 性能统计
    t0 = time.perf_counter()
    sum_full = 0.0
    sum_obb = 0.0
    sum_pose = 0.0
    frames_done = 0
    full_calls = 0

    fps_window = deque(maxlen=60)
    last_tick = time.perf_counter()

    # 额外统计
    last_xywhr = None

    try:
        for i in range(1, args.frames + 1):
            frame_rgb = env.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            show = frame_bgr.copy()

            tt_full0 = time.perf_counter()

            did_full = False
            xywhr = None

            if (i % args.freq) == 1:  # 每 N 帧完整 OBB+Pose
                did_full = True
                # OBB+Pose 由 est.update_full 内部完成（带角度约束）
                tt_obb0 = time.perf_counter()
                out = est.update_full(frame_bgr)
                torch_sync(device)
                tt_obb1 = time.perf_counter()

                # 为了分开计时 OBB/Pose，你可以在 FusedStateEstimator 里拆 timing；这里只统计总的 full
                sum_obb += (tt_obb1 - tt_obb0)  # 这里其实是 full 的时间（近似）
                if out is not None:
                    # 只用于画 OBB（可选）：在 _infer_once 中拿到的 OBB 结构没有对外暴露，
                    # 这里用 last_xywhr 代替（若想精准，可在类里保存最后一次 OBB xywhr 对外提供）。
                    pass
            else:
                out = est.predict_only()
                torch_sync(device)

            tt_full1 = time.perf_counter()
            sum_full += (tt_full1 - tt_full0)
            if did_full:
                full_calls += 1

            # FPS
            now = time.perf_counter()
            dt = now - last_tick
            last_tick = now
            if dt > 0:
                fps_window.append(1.0/dt)
            fps = np.mean(fps_window) if fps_window else 0.0

            # 统计
            stats = est.get_debug_stats()
            hard_rej = stats["hard_rejects"]
            soft_clp = stats["soft_clamps"]

            if args.visualize:
                msg1 = f"Frame {i}/{args.frames}  |  FPS: {fps:5.1f}"
                msg2 = f"Full(ms): {(sum_full/max(1,frames_done+1))*1000:6.1f}  FullCalls: {full_calls}"
                msg3 = f"HardRejects: {hard_rej}  SoftClamps: {soft_clp}"
                cv2.putText(show, msg1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2, cv2.LINE_AA)
                cv2.putText(show, msg2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(show, msg3, (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 180, 255), 2, cv2.LINE_AA)
                cv2.imshow("speed_check (PT + CUDA) with angle constraints", show)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            # 环境步进（随机动作，仅为刷新画面）
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
                est.begin_episode()

            frames_done += 1

    finally:
        env.close()
        cv2.destroyAllWindows()

    t1 = time.perf_counter()
    total = t1 - t0
    print(f"\nFrames: {frames_done}, Time: {total:.2f}s, ~{frames_done/max(1,total):.2f} FPS")
    if full_calls:
        print(f"Avg Full/frame: {(sum_full/max(1,frames_done))*1000:.1f} ms")
    print("Angle-limit stats:", est.get_debug_stats())

if __name__ == "__main__":
    main()
