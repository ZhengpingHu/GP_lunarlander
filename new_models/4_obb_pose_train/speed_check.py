#!/usr/bin/env python3
# speed_check.py  (Windows + CUDA, PT only)
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 防止 Windows 上 OpenMP 重复加载报错
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # 如需强制CPU可改成 "":CPU; 否则下面会自动选CUDA

import time
import math
import argparse
from collections import deque

import numpy as np
import cv2
import torch
from ultralytics import YOLO
import gymnasium as gym

# ------------------ 工具函数 ------------------

def torch_sync(device):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

def affine_and_crop_bgr(frame_rgb, cx, cy, w, h, theta_rad, pad_ratio):
    """
    根据 OBB 旋转并裁剪出姿态区域（BGR）。
    """
    H, W = frame_rgb.shape[:2]
    angle_deg = -theta_rad * 180.0 / math.pi  # OpenCV 旋转角度为逆时针，屏幕y向下，取负号
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

def draw_obb_on(frame, xywhr, color=(0,255,255), thickness=2):
    """
    仅用于可视化：把 xywhr 画到图上。
    xywhr: (cx, cy, w, h, theta)
    """
    if xywhr is None:
        return frame
    cx, cy, w, h, th = xywhr
    c = np.array([cx, cy], dtype=np.float32)
    # 旋转矩形四点
    cos_t = math.cos(th); sin_t = math.sin(th)
    dx = w/2.0; dy = h/2.0
    pts = np.array([
        [-dx, -dy],
        [ dx, -dy],
        [ dx,  dy],
        [-dx,  dy]
    ], dtype=np.float32)
    R = np.array([[cos_t, -sin_t],[sin_t, cos_t]], dtype=np.float32)
    pts = (pts @ R.T) + c
    pts = pts.astype(int)
    cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)
    # 画中心
    cv2.circle(frame, (int(cx),int(cy)), 3, (0,0,255), -1, cv2.LINE_AA)
    return frame

# ------------------ 主逻辑 ------------------

def main():
    ap = argparse.ArgumentParser(description="YOLO OBB+Pose speed check (Windows+CUDA, PT-only)")
    ap.add_argument("--obb", default="./lander.pt", help="Path to OBB .pt")
    ap.add_argument("--pose", default="./lander-pose.pt", help="Path to Pose .pt")
    ap.add_argument("--imgsz-obb", type=int, default=640, help="OBB inference size (must be what the model expects)")
    ap.add_argument("--imgsz-pose", type=int, default=384, help="Pose inference size (must be what the model expects)")
    ap.add_argument("--conf-obb", type=float, default=0.25, help="OBB conf")
    ap.add_argument("--conf-pose", type=float, default=0.20, help="Pose conf")
    ap.add_argument("--pad", type=float, default=0.25, help="crop padding ratio around OBB")
    ap.add_argument("--frames", type=int, default=400, help="Total frames to run")
    ap.add_argument("--freq", type=int, default=10, help="Do full OBB+Pose every N frames (frame-skip)")
    ap.add_argument("--visualize", action="store_true", help="Show cv2 window")
    ap.add_argument("--device", default="auto", choices=["auto","cuda","cpu"], help="Preferred device")
    args = ap.parse_args()

    # 设备选择
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[警告] 选择了 CUDA 但不可用，退回 CPU")
        device = "cpu"
    elif args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"[信息] 使用设备: {device}")

    # 加载模型（.pt）
    print(f"[信息] 加载 OBB 模型: {args.obb}")
    obb_model = YOLO(args.obb)  # task 自动推断
    print(f"[信息] 加载 Pose 模型: {args.pose}")
    pose_model = YOLO(args.pose)

    # Gym 环境
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    obs, _ = env.reset()

    # 性能统计
    t0 = time.perf_counter()
    sum_full = 0.0        # 全流程时延（包含渲染、预处理、OBB/POSE）
    sum_obb = 0.0         # OBB 推理时延
    sum_pose = 0.0        # Pose 推理时延
    frames_done = 0
    full_calls = 0

    # 显示用的滑窗 FPS
    fps_window = deque(maxlen=60)
    last_tick = time.perf_counter()

    try:
        for i in range(1, args.frames + 1):
            frame_rgb = env.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            show = frame_bgr.copy()

            tt_full0 = time.perf_counter()

            did_full = False
            xywhr = None

            if (i % args.freq) == 1:  # 每N帧运行一次完整 OBB+Pose
                did_full = True
                # ---------- OBB ----------
                tt_obb0 = time.perf_counter()
                obb_r = obb_model(frame_bgr, imgsz=args.imgsz_obb, conf=args.conf_obb, verbose=False)[0]
                torch_sync(device)
                tt_obb1 = time.perf_counter()
                sum_obb += (tt_obb1 - tt_obb0)

                has_obb = (obb_r.obb is not None) and (obb_r.obb.xywhr is not None) and (len(obb_r.obb.xywhr) > 0)
                if has_obb:
                    idx = int(obb_r.obb.conf.argmax().cpu().item())
                    cx, cy, w, h, theta_obb = obb_r.obb.xywhr[idx].cpu().numpy().tolist()
                    xywhr = (cx, cy, w, h, theta_obb)

                    # ---------- 裁剪 + Pose ----------
                    crop_bgr, meta = affine_and_crop_bgr(frame_rgb, cx, cy, w, h, theta_obb, pad_ratio=args.pad)
                    if crop_bgr is not None and crop_bgr.size > 0:
                        tt_pose0 = time.perf_counter()
                        pose_r = pose_model(crop_bgr, imgsz=args.imgsz_pose, conf=args.conf_pose, verbose=False)[0]
                        torch_sync(device)
                        tt_pose1 = time.perf_counter()
                        sum_pose += (tt_pose1 - tt_pose0)

                        if args.visualize:
                            # 简单画一下 keypoints
                            if pose_r.keypoints is not None and getattr(pose_r.keypoints, "xy", None) is not None and len(pose_r.keypoints.xy) > 0:
                                kp = pose_r.keypoints.xy[0].cpu().numpy()
                                # 映射回原图
                                M, x0c, y0c = meta
                                k_rot = kp.copy()
                                k_rot[:, 0] += x0c
                                k_rot[:, 1] += y0c
                                inv_M = cv2.invertAffineTransform(M)
                                k_homo = np.hstack([k_rot, np.ones((k_rot.shape[0], 1), dtype=np.float32)])
                                k_org = (inv_M @ k_homo.T).T
                                for (xk, yk) in k_org:
                                    cv2.circle(show, (int(xk), int(yk)), 3, (0, 255, 0), -1, cv2.LINE_AA)

                # OBB 可视化
                if args.visualize and xywhr is not None:
                    show = draw_obb_on(show, xywhr, color=(0, 255, 255), thickness=2)

            # 统计与显示
            torch_sync(device)
            tt_full1 = time.perf_counter()
            sum_full += (tt_full1 - tt_full0)
            if did_full:
                full_calls += 1

            # 统计 FPS
            now = time.perf_counter()
            dt = now - last_tick
            last_tick = now
            if dt > 0:
                fps_window.append(1.0/dt)
            fps = np.mean(fps_window) if fps_window else 0.0

            if args.visualize:
                h, w = show.shape[:2]
                msg1 = f"Frame {i}/{args.frames}  |  FPS: {fps:5.1f}"
                msg2 = f"Full(ms): { (sum_full/max(1, i))*1000:6.1f}  OBB(ms): { (sum_obb/max(1, full_calls))*1000:6.1f}  Pose(ms): { (sum_pose/max(1, full_calls))*1000:6.1f}"
                cv2.putText(show, msg1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2, cv2.LINE_AA)
                cv2.putText(show, msg2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("speed_check (PT + CUDA)", show)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            # 环境步进（随机动作，仅为刷新画面）
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

            frames_done += 1

    finally:
        env.close()
        cv2.destroyAllWindows()

    t1 = time.perf_counter()
    total = t1 - t0
    print(f"\nFrames: {frames_done}, Time: {total:.2f}s, ~{frames_done/max(1,total):.2f} FPS")
    if full_calls:
        print(f"Avg Full/frame: {(sum_full/frames_done)*1000:.1f} ms | "
              f"Avg OBB(full-call): {(sum_obb/full_calls)*1000:.1f} ms | "
              f"Avg Pose(full-call): {(sum_pose/full_calls)*1000:.1f} ms")

if __name__ == "__main__":
    main()
