#!/usr/bin/env python3
# yolo_lander_analysis_fused.py — OBB vs Pose vs Fused（鲁棒测角=脚尖+PCA兜底 / 自适应pad / 门控 / 平滑 / 自动reset / 图表+CSV）

import argparse, os, sys, math
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import gymnasium as gym

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"


# ---------- helpers ----------
def wrap_pi(a: float) -> float:
    while a <= -math.pi: a += 2*math.pi
    while a >  math.pi: a -= 2*math.pi
    return a

def ang_diff(a: float, b: float) -> float:
    return abs(wrap_pi(a - b))

def ang_axis_diff(a: float, b: float) -> float:
    d = ang_diff(a, b)
    return min(d, abs(wrap_pi(d - math.pi)))

def rad2deg(r: float) -> float:
    return r * 180.0 / math.pi

def robust_angle_from_kpts(k_org, use_pca_fallback=True, min_dx_px=3.0):
    """
    仅用 脚尖(left->right) + PCA 兜底 的鲁棒测角（不再使用屋顶边，避免0/1顺序歧义）
    k_org: (K,2)，至少前4点分别为 [*, *, left_foot, right_foot]
    返回: 弧度（图像坐标，y向下），或 None
    """
    if k_org is None or k_org.shape[0] < 4 or not np.all(np.isfinite(k_org[:4])):
        return None

    feet = k_org[2:4].astype(np.float32).copy()
    if feet.shape[0] < 2 or not np.all(np.isfinite(feet)):
        return None
    order = np.argsort(feet[:, 0])
    if order.size < 2:
        return None
    left_foot, right_foot = feet[order[0]], feet[order[1]]

    dx = float(right_foot[0] - left_foot[0])
    dy = float(right_foot[1] - left_foot[1])
    base_dx = abs(dx)

    if base_dx >= min_dx_px:
        return wrap_pi(math.atan2(dy, dx))

    if not use_pca_fallback:
        return wrap_pi(math.atan2(dy, dx))

    P = k_org[:4, :2].astype(np.float32)
    if not np.all(np.isfinite(P)):
        return wrap_pi(math.atan2(dy, dx))

    mu = P.mean(axis=0, keepdims=True)
    X = P - mu
    C = (X.T @ X) / max(len(P) - 1, 1)
    eigvals, eigvecs = np.linalg.eig(C)
    v = eigvecs[:, int(np.argmax(eigvals))]

    leftmost  = P[np.argmin(P[:, 0])]
    rightmost = P[np.argmax(P[:, 0])]
    ref = (rightmost - leftmost).astype(np.float32)
    if np.dot(v, ref) < 0:
        v = -v

    theta = math.atan2(float(v[1]), float(v[0]))
    return wrap_pi(theta)

def affine_and_crop_bgr(frame_rgb: np.ndarray, cx: float, cy: float, w: float, h: float,
                        theta_rad: float, pad_ratio: float):
    H, W = frame_rgb.shape[:2]
    angle_deg = -theta_rad * 180.0 / math.pi
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
    return cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR).copy(), (M, x0c, y0c)

def near_vertical(theta, thresh_deg=20.0):
    d = min(abs(wrap_pi(theta - math.pi/2)), abs(wrap_pi(theta + math.pi/2)))
    return d < math.radians(thresh_deg)

def suggest_conf_threshold(df_conf_err: pd.DataFrame, err_key: str, conf_key: str,
                           target_rate=0.95, target_err=2.0) -> float:
    best = 0.70
    for t in np.arange(0.70, 1.001, 0.01):
        sub = df_conf_err[df_conf_err[conf_key] >= t]
        if len(sub) >= 50:
            rate = (sub[err_key] < target_err).mean()
            if rate >= target_rate:
                best = float(round(t, 2))
                break
    return best

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Lander OBB vs Pose vs Fused analysis (robust angle)")
    ap.add_argument("--obb-model", required=True)
    ap.add_argument("--pose-model", required=True)
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--obb-conf", type=float, default=0.25)
    ap.add_argument("--pose-conf", type=float, default=0.20)
    ap.add_argument("--pad", type=float, default=0.25)
    ap.add_argument("--gate-deg", type=float, default=120.0)
    ap.add_argument("--smooth-alpha", type=float, default=0.2)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--outdir", default="analysis_results_fused")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("[信息] 加载模型...")
    try:
        obb = YOLO(args.obb_model, task="obb")
        pose = YOLO(args.pose_model, task="pose")
        print("[信息] 模型加载成功。")
    except Exception as e:
        print(f"[错误] 加载失败: {e}")
        sys.exit(1)

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset()

    gate_rad = math.radians(args.gate_deg)
    prev_theta = None
    rows = []
    episode = 1
    print(f"[信息] Episode #{episode} 开始")
    print(f"[信息] 运行 {args.frames} 帧采样并记录 (OBB / Pose / Fused)... 按 'q' 可中断可视化。")

    for i in range(1, args.frames + 1):
        frame = env.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # True angle（世界坐标->图像坐标）
        try:
            true_theta_world = float(env.unwrapped.lander.angle)
        except Exception:
            true_theta_world = float(obs[4])
        true_theta = -true_theta_world

        # ---------- OBB ----------
        r = obb(frame_bgr, imgsz=args.imgsz, conf=args.obb_conf, verbose=False)[0]
        obb_ok = False
        cx = cy = w = h = theta_obb = None
        obb_conf = float("nan")

        if hasattr(r, "obb") and r.obb is not None and r.obb.xywhr is not None and len(r.obb.xywhr):
            idx = int(r.obb.conf.argmax().cpu().item())
            cx, cy, w, h, th = r.obb.xywhr[idx].cpu().numpy().tolist()
            theta_obb = float(th)
            obb_conf = float(r.obb.conf[idx].cpu().item())
            obb_ok = True
        elif r.boxes is not None and len(r.boxes):
            idx = int(r.boxes.conf.argmax().cpu().item())
            x, y, w, h = r.boxes.xywh[idx].cpu().numpy().tolist()
            cx, cy, theta_obb = x, y, 0.0
            obb_conf = float(r.boxes.conf[idx].cpu().item())
            obb_ok = True

        pose_ok = False
        theta_pose = None
        pose_det_conf = float("nan")

        if obb_ok:
            # 自适应 pad
            pad_local = args.pad
            if theta_obb is not None and near_vertical(theta_obb, 20.0):
                pad_local = max(pad_local, 0.30)

            crop_bgr, meta = affine_and_crop_bgr(frame, cx, cy, w, h, theta_obb, pad_ratio=pad_local)
            if crop_bgr is not None and crop_bgr.size > 0:
                pr = pose(crop_bgr, imgsz=args.imgsz, conf=args.pose_conf, verbose=False)[0]
                has_det = (pr.boxes is not None) and (len(pr.boxes) > 0)
                has_k = (pr.keypoints is not None) and (getattr(pr.keypoints, "xy", None) is not None) and (len(pr.keypoints.xy) > 0)
                if has_det and has_k:
                    det_idx = int(pr.boxes.conf.argmax().cpu().item())
                    kp_list = pr.keypoints.xy
                    if det_idx >= len(kp_list):
                        det_idx = 0
                    k = kp_list[det_idx].cpu().numpy()
                    if k.ndim == 2 and k.shape[0] >= 4 and np.all(np.isfinite(k)):
                        M, x0c, y0c = meta
                        k_rot = k.copy()
                        k_rot[:, 0] += x0c
                        k_rot[:, 1] += y0c
                        inv_M = cv2.invertAffineTransform(M)
                        k_homo = np.hstack([k_rot, np.ones((k_rot.shape[0], 1), dtype=np.float32)])
                        k_org = (inv_M @ k_homo.T).T  # (K,2)

                        if k_org.shape[0] >= 4 and np.all(np.isfinite(k_org[:4])):
                            theta_pose = robust_angle_from_kpts(k_org, use_pca_fallback=True, min_dx_px=3.0)
                            pose_det_conf = float(pr.boxes.conf[det_idx].cpu().item())
                            pose_ok = (theta_pose is not None)

        # ---------- Fusion（分析：Pose角为主，门控+平滑） ----------
        theta_fused = None
        if pose_ok:
            theta_fused = theta_pose
            if not args.no_gate and obb_ok and ang_axis_diff(theta_pose, theta_obb) > gate_rad:
                theta_fused = prev_theta if prev_theta is not None else theta_obb
        elif obb_ok:
            theta_fused = theta_obb

        if (not args.no_smooth) and (theta_fused is not None) and (prev_theta is not None):
            alpha = float(np.clip(args.smooth_alpha, 0.0, 1.0))
            theta_fused = wrap_pi(alpha*theta_fused + (1.0-alpha)*prev_theta)
        if theta_fused is not None:
            prev_theta = theta_fused
        else:
            prev_theta = None

        # ---------- Metrics ----------
        def err_deg(pred_rad):
            if pred_rad is None: return float("nan")
            return rad2deg(ang_diff(pred_rad, true_theta))

        row = {
            "frame": i,
            "true_deg": rad2deg(true_theta),
            "obb_deg": rad2deg(theta_obb) if theta_obb is not None else float("nan"),
            "pose_deg": rad2deg(theta_pose) if theta_pose is not None else float("nan"),
            "fused_deg": rad2deg(theta_fused) if theta_fused is not None else float("nan"),
            "err_obb_deg": err_deg(theta_obb),
            "err_pose_deg": err_deg(theta_pose),
            "err_fused_deg": err_deg(theta_fused),
            "flip_obb": 1 if (theta_obb is not None and ang_diff(theta_obb, true_theta) > math.pi/2) else 0,
            "flip_pose": 1 if (theta_pose is not None and ang_diff(theta_pose, true_theta) > math.pi/2) else 0,
            "flip_fused":1 if (theta_fused is not None and ang_diff(theta_fused, true_theta) > math.pi/2) else 0,
            "obb_conf": obb_conf,
            "pose_det_conf": pose_det_conf
        }
        rows.append(row)

        # ---------- Live viz ----------
        if args.visualize:
            disp = frame.copy()
            txt = f"T:{row['true_deg']:+.1f} | O:{row['obb_deg'] if not np.isnan(row['obb_deg']) else None} | P:{row['pose_deg'] if not np.isnan(row['pose_deg']) else None} | F:{row['fused_deg'] if not np.isnan(row['fused_deg']) else None}"
            cv2.putText(disp, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,255,50), 2, cv2.LINE_AA)
            cv2.imshow("Analysis Live", cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # ---------- Env step + auto reset ----------
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            episode += 1
            obs, _ = env.reset()
            prev_theta = None
            if args.visualize:
                print(f"\n[信息] Episode #{episode} 开始")

        if i % 50 == 0:
            print(f"\r[进度] {i}/{args.frames}", end="")

    env.close()
    cv2.destroyAllWindows()
    print("\n[信息] 采样完成，生成报表...")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.outdir, "fused_report.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[信息] CSV 已保存：{csv_path}")

    # ---------- Summary ----------
    def summarize(tag, key_err, key_flip):
        sub = df[np.isfinite(df[key_err])]
        mae = sub[key_err].mean() if len(sub) else float("nan")
        flip = sub[key_flip].mean()*100.0 if len(sub) else float("nan")
        n = len(sub)
        print(f"[{tag}] Angle MAE: {mae:.3f} deg | Flip Rate: {flip:.2f}% | N={n}")
        return mae, flip, n

    summarize("OBB ", "err_obb_deg",  "flip_obb")
    summarize("POSE", "err_pose_deg", "flip_pose")
    summarize("FUSE", "err_fused_deg","flip_fused")

    # ---------- Plots ----------
    os.makedirs(args.outdir, exist_ok=True)

    # 1) error vs true angle bins
    for key_err, name in [("err_obb_deg","OBB"), ("err_pose_deg","POSE"), ("err_fused_deg","FUSED")]:
        sub = df[np.isfinite(df[key_err])]
        if len(sub) == 0:
            continue
        sub = sub.copy()
        sub["angle_bin"] = sub["true_deg"].round().astype(int)
        gb = sub.groupby("angle_bin")[key_err].mean()
        plt.figure(figsize=(14,6))
        gb.plot(kind="bar", color="skyblue")
        plt.title(f"[{name}] Average Absolute Error vs True Angle")
        plt.xlabel("True Angle (deg)")
        plt.ylabel("Average Abs Error (deg)")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        ax = plt.gca()
        ticks = ax.get_xticks(); labels = ax.get_xticklabels()
        n = max(1, len(ticks)//20)
        ax.set_xticks(ticks[::n]); ax.set_xticklabels(labels[::n])
        outp = os.path.join(args.outdir, f"error_vs_true_{name.lower()}.png")
        plt.savefig(outp); plt.close()
        print(f"[图] {outp}")

    # 2) error vs confidence (scatter)
    # OBB
    subo = df[np.isfinite(df["err_obb_deg"]) & np.isfinite(df["obb_conf"])]
    if len(subo):
        plt.figure(figsize=(10,6))
        plt.scatter(subo["obb_conf"], subo["err_obb_deg"], alpha=0.4)
        plt.axhline(2.0, color="r", linestyle="--", label="2° threshold")
        plt.title("OBB: Absolute Error vs Confidence")
        plt.xlabel("OBB Confidence"); plt.ylabel("Abs Error (deg)")
        plt.grid(True, linestyle="--", alpha=0.6); plt.legend()
        outp = os.path.join(args.outdir, "error_vs_conf_obb.png")
        plt.savefig(outp); plt.close(); print(f"[图] {outp}")

    # Pose
    subp = df[np.isfinite(df["err_pose_deg"]) & np.isfinite(df["pose_det_conf"])]
    if len(subp):
        plt.figure(figsize=(10,6))
        plt.scatter(subp["pose_det_conf"], subp["err_pose_deg"], alpha=0.4)
        plt.axhline(2.0, color="r", linestyle="--", label="2° threshold")
        plt.title("POSE: Absolute Error vs Detection Confidence")
        plt.xlabel("Pose Detection Confidence"); plt.ylabel("Abs Error (deg)")
        plt.grid(True, linestyle="--", alpha=0.6); plt.legend()
        outp = os.path.join(args.outdir, "error_vs_conf_pose.png")
        plt.savefig(outp); plt.close(); print(f"[图] {outp}")

    # 3) 建议阈值（95% < 2°）
    if len(subo):
        thr_obb = suggest_conf_threshold(subo.rename(columns={"err_obb_deg":"err","obb_conf":"conf"}), "err", "conf")
        print(f"[建议] OBB置信度阈值：{thr_obb:.2f} (95%样本<2°)")
    if len(subp):
        thr_pose = suggest_conf_threshold(subp.rename(columns={"err_pose_deg":"err","pose_det_conf":"conf"}), "err", "conf")
        print(f"[建议] POSE置信度阈值：{thr_pose:.2f} (95%样本<2°)")

    print("--- 报告生成完毕 ---")

if __name__ == "__main__":
    main()
