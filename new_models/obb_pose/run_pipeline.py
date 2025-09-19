#!/usr/bin/env python3
# run_pipeline.py — Pose优先 + 置信度兜底到OBB（鲁棒测角=脚尖+PCA兜底 / 门控 / 平滑 / 自适应pad / 自动reset）

import argparse, sys, math
import numpy as np
import cv2
from ultralytics import YOLO
import gymnasium as gym

# ---------- helpers ----------
def wrap_pi(a: float) -> float:
    while a <= -math.pi: a += 2 * math.pi
    while a >  math.pi: a -= 2 * math.pi
    return a

def ang_diff(a: float, b: float) -> float:
    """有方向差 |a-b| 归一到 [-pi, pi] 后取绝对值"""
    return abs(wrap_pi(a - b))

def ang_axis_diff(a: float, b: float) -> float:
    """
    无方向（轴向）差：min(|a-b|, |a-b±pi|) ∈ [0, pi/2]
    OBB 角通常无方向，用此比较可避免 ±180° 误触发门控
    """
    d = ang_diff(a, b)
    return min(d, abs(wrap_pi(d - math.pi)))

def robust_angle_from_kpts(k_org, use_pca_fallback=True, min_dx_px=3.0):
    """
    仅用 脚尖(left->right) + PCA 兜底 的鲁棒测角（不再使用屋顶边，避免0/1顺序歧义）
    k_org: (K,2)，至少前4点分别为 [*, *, left_foot, right_foot]
    返回: 弧度（图像坐标，y向下），或 None
    """
    if k_org is None or k_org.shape[0] < 4 or not np.all(np.isfinite(k_org[:4])):
        return None

    # 取脚尖（index 2,3），按 x 排序保证 left->right
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

    # 水平跨度足够时直接用脚尖连线
    if base_dx >= min_dx_px:
        theta = math.atan2(dy, dx)
        return wrap_pi(theta)

    # 否则 PCA 兜底（用前4点）
    if not use_pca_fallback:
        return wrap_pi(math.atan2(dy, dx))

    P = k_org[:4, :2].astype(np.float32)
    if not np.all(np.isfinite(P)):
        return wrap_pi(math.atan2(dy, dx))

    mu = P.mean(axis=0, keepdims=True)
    X = P - mu
    C = (X.T @ X) / max(len(P) - 1, 1)
    eigvals, eigvecs = np.linalg.eig(C)
    v = eigvecs[:, int(np.argmax(eigvals))]  # 主方向（x,y）

    # 方向消歧：让主方向与“全体点里(最右-最左)”同向，避免 ±pi 翻转
    leftmost  = P[np.argmin(P[:, 0])]
    rightmost = P[np.argmax(P[:, 0])]
    ref = (rightmost - leftmost).astype(np.float32)
    if np.dot(v, ref) < 0:
        v = -v

    theta = math.atan2(float(v[1]), float(v[0]))
    return wrap_pi(theta)

def affine_and_crop_bgr(frame_rgb, cx, cy, w, h, theta_rad, pad_ratio):
    """
    按 -theta 旋正并在旋正图上裁剪，返回 BGR 裁剪图和 (M, x0c, y0c)
    M: 原图 -> 旋正图 的仿射
    (x0c, y0c): 裁剪框左上角在旋正图坐标
    """
    H, W = frame_rgb.shape[:2]
    angle_deg = -theta_rad * 180.0 / math.pi
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rotated = cv2.warpAffine(frame_rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    w_pad = w * (1.0 + 2.0 * pad_ratio)
    h_pad = h * (1.0 + 2.0 * pad_ratio)
    x0 = int(round(cx - w_pad / 2)); y0 = int(round(cy - h_pad / 2))
    x1 = int(round(cx + w_pad / 2)); y1 = int(round(cy + h_pad / 2))
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x1), min(H, y1)
    if x1c <= x0c or y1c <= y0c:
        return None, None
    crop_rgb = rotated[y0c:y1c, x0c:x1c].copy()
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR).copy()  # YOLO 推理用 BGR
    return crop_bgr, (M, x0c, y0c)

def near_vertical(theta, thresh_deg=20.0):
    """判断角度是否接近竖直（±90°附近）"""
    d = min(abs(wrap_pi(theta - math.pi/2)), abs(wrap_pi(theta + math.pi/2)))
    return d < math.radians(thresh_deg)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Two-Stage Lander Pipeline (Pose-first with confidence fallback to OBB)")
    ap.add_argument("--obb-model", required=True, help="Stage-1 OBB model path")
    ap.add_argument("--pose-model", required=True, help="Stage-2 Pose model path")
    ap.add_argument("--conf", type=float, default=0.25, help="OBB conf threshold")
    ap.add_argument("--pose-conf", type=float, default=0.20, help="Pose detection conf threshold (for running)")
    ap.add_argument("--pose-min-conf", type=float, default=0.94, help="Pose结果用于角度前的最小置信度阈值（兜底到OBB）")
    ap.add_argument("--imgsz", type=int, default=512, help="img size for both models")
    ap.add_argument("--pad", type=float, default=0.25, help="base padding ratio for crop")
    ap.add_argument("--gate-deg", type=float, default=120.0, help="axis gate in degrees")
    ap.add_argument("--smooth-alpha", type=float, default=0.2, help="IIR smoothing alpha [0..1]")
    ap.add_argument("--no-gate", action="store_true", help="disable axis gating")
    ap.add_argument("--no-smooth", action="store_true", help="disable temporal smoothing")
    args = ap.parse_args()

    print("[信息] 正在加载模型...")
    try:
        obb_model = YOLO(args.obb_model, task="obb")
        pose_model = YOLO(args.pose_model, task="pose")
        print("[信息] 模型加载成功！")
    except Exception as e:
        print(f"[错误] 加载模型失败: {e}")
        sys.exit(1)

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset()
    print("[信息] 流水线启动。按 'q' 退出。")

    prev_theta = None
    gate_rad = math.radians(args.gate_deg)
    episode = 1
    print(f"[信息] Episode #{episode} 开始")

    while True:
        frame_rgb = env.render()
        display_frame = frame_rgb.copy()

        # -------- Stage 1: OBB on full frame --------
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        r = obb_model(frame_bgr, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

        has_obb = False
        cx = cy = w = h = theta_obb = None
        if r.obb is not None and r.obb.xywhr is not None and len(r.obb.xywhr):
            idx = int(r.obb.conf.argmax().cpu().item())
            cx, cy, w, h, theta_obb = r.obb.xywhr[idx].cpu().numpy().tolist()
            has_obb = True
        elif r.boxes is not None and len(r.boxes):
            idx = int(r.boxes.conf.argmax().cpu().item())
            x, y, w, h = r.boxes.xywh[idx].cpu().numpy().tolist()
            cx, cy, theta_obb = x, y, 0.0
            has_obb = True

        if not has_obb:
            cv2.imshow("Pipeline", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                episode += 1
                obs, _ = env.reset()
                prev_theta = None
                print(f"\n[信息] Episode #{episode} 开始")
            continue

        # 可视化 OBB 中心
        cv2.circle(display_frame, (int(cx), int(cy)), 3, (255, 255, 0), -1)

        # -------- 自适应 pad：接近竖直时增大裁剪留量 --------
        pad_local = args.pad
        if theta_obb is not None and near_vertical(theta_obb, 20.0):
            pad_local = max(pad_local, 0.30)

        # -------- Stage 2: rectify & crop, Pose on crop --------
        crop_bgr, meta = affine_and_crop_bgr(frame_rgb, cx, cy, w, h, theta_obb, pad_ratio=pad_local)
        if crop_bgr is None or crop_bgr.size == 0:
            cv2.imshow("Pipeline", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                episode += 1
                obs, _ = env.reset()
                prev_theta = None
                print(f"\n[信息] Episode #{episode} 开始")
            continue

        pr = pose_model(crop_bgr, imgsz=args.imgsz, conf=args.pose_conf, verbose=False)[0]

        # ---- 可靠性检查 ----
        use_pose = False
        theta_pose = None
        pose_det_conf = float("nan")

        if (pr.boxes is not None and len(pr.boxes) > 0) and (pr.keypoints is not None) and (getattr(pr.keypoints, "xy", None) is not None) and (len(pr.keypoints.xy) > 0):
            det_idx = int(pr.boxes.conf.argmax().cpu().item())
            kp_list = pr.keypoints.xy
            if det_idx >= len(kp_list):
                det_idx = 0
            k = kp_list[det_idx].cpu().numpy()
            if k.ndim == 2 and k.shape[0] >= 4 and np.all(np.isfinite(k)):
                # map crop -> original
                M, x0c, y0c = meta
                k_rot = k.copy()
                k_rot[:, 0] += x0c
                k_rot[:, 1] += y0c
                inv_M = cv2.invertAffineTransform(M)
                k_homo = np.hstack([k_rot, np.ones((k_rot.shape[0], 1), dtype=np.float32)])
                k_org = (inv_M @ k_homo.T).T  # (K,2)

                if k_org.shape[0] >= 4 and np.all(np.isfinite(k_org[:4])):
                    # 画点（前4个）
                    H, W = frame_rgb.shape[:2]
                    for i, (gx, gy) in enumerate(k_org[:4]):
                        if 0 <= gx < W and 0 <= gy < H:
                            cv2.circle(display_frame, (int(gx), int(gy)), 4, (0, 255, 0), -1)
                            cv2.putText(display_frame, str(i), (int(gx)+5, int(gy)+5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                    # 鲁棒测角（原图）
                    theta_pose = robust_angle_from_kpts(k_org, use_pca_fallback=True, min_dx_px=3.0)
                    pose_det_conf = float(pr.boxes.conf[det_idx].cpu().item())
                    use_pose = (theta_pose is not None) and (pose_det_conf >= args.pose_min_conf)

        # ---- 决策：Pose优先 + 兜底到 OBB ----
        if use_pose and theta_pose is not None:
            theta_candidate = theta_pose
            source = f"POSE ({pose_det_conf:.2f} ≥ {args.pose_min_conf:.2f})"
            # 轴向门控（可选）
            if (not args.no_gate) and (theta_obb is not None) and ang_axis_diff(theta_pose, theta_obb) > math.radians(args.gate_deg):
                theta_candidate = theta_obb  # 门控驳回，使用 OBB
                source = f"OBB [GATED] ({pose_det_conf:.2f})"
        else:
            theta_candidate = theta_obb
            source = f"OBB (pose<{args.pose_min_conf:.2f} or fail)"

        # ---- 时序平滑（可选）----
        theta = theta_candidate
        if (not args.no_smooth) and (prev_theta is not None) and (theta is not None):
            alpha = float(np.clip(args.smooth_alpha, 0.0, 1.0))
            theta = wrap_pi(alpha * theta + (1.0 - alpha) * prev_theta)
            source += " + SMOOTH"
        prev_theta = theta

        # ---- 文本显示 ----
        txt = []
        if theta_pose is not None:
            txt.append(f"Pose:{theta_pose*180/math.pi:+.1f}")
            txt.append(f"conf:{pose_det_conf:.2f}")
        if theta_obb is not None:
            txt.append(f"OBB:{theta_obb*180/math.pi:+.1f}")
        if theta is not None:
            txt.append(f"Fused:{theta*180/math.pi:+.1f}")
        txt.append(f"[{source}]")
        cv2.putText(display_frame, "  ".join(txt), (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (50,255,50), 2, cv2.LINE_AA)

        cv2.imshow("Pipeline", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # -------- 环境步进 + 自动 reset --------
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            episode += 1
            obs, _ = env.reset()
            prev_theta = None
            print(f"\n[信息] Episode #{episode} 开始")

    env.close()
    cv2.destroyAllWindows()
    print("\n[信息] 流水线已关闭。")

if __name__ == "__main__":
    main()
