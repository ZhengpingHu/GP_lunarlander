#!/usr/bin/env python3
# prepare_pose_from_obb.py
import os, math, glob, json
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO

# -------------------------
# 可配置
# -------------------------
FULL_IMG_DIR = "training_data_pose/images"
FULL_LBL_DIR = "training_data_pose/labels"
OUT_IMG_DIR  = "crops_from_obb/images/train"   # 先全量放 train，之后你再切分
OUT_LBL_DIR  = "crops_from_obb/labels/train"
OBB_MODEL = r"D:\Git\GP_lunarlander\new_models\obb_pose\lander.pt"
PAD_RATIO    = 0.15                # 在 w,h 上各加 15% 边界
MIN_CONF     = 0.25                # 过滤低置信度预测
CLASS_ID     = 0                   # lander 类别id（按你的模型设定）

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

# -------------------------
# 读你原有的 YOLO Pose 标签（与 generate_pose_data.py 输出一致）
# 格式: cls cx cy w h k1x k1y k2x k2y k3x k3y k4x k4y （均为归一化，相对大图宽高）
# -------------------------
def read_full_label(label_path, W, H, num_kpts=4):
    with open(label_path, "r") as f:
        line = f.read().strip().split()
    cls = int(float(line[0]))
    cx, cy, bw, bh = map(float, line[1:5])
    # 关键点（归一化） -> 像素
    arr = list(map(float, line[5:5+2*num_kpts]))
    kpts = np.array(arr, dtype=np.float32).reshape(num_kpts, 2)
    kpts_px = np.zeros_like(kpts)
    kpts_px[:, 0] = kpts[:, 0] * W
    kpts_px[:, 1] = kpts[:, 1] * H
    return cls, (cx*W, cy*H, bw*W, bh*H), kpts_px

# -------------------------
# 将 OBB 结果读取为 (cx, cy, w, h, theta_rad)
# Ultralytics OBB 可能提供 xywhr 或多边形；都兼容一下
# -------------------------
def extract_obb_xywhr(result):
    # 优先：xywhr
    if hasattr(result, "obb") and hasattr(result.obb, "xywhr") and result.obb.xywhr is not None:
        xywhr = result.obb.xywhr.cpu().numpy()        # (N,5): cx,cy,w,h,theta(rad)
        conf  = result.obb.conf.cpu().numpy()         # (N,)
        cls   = result.obb.cls.cpu().numpy()          # (N,)
        polys = None
    # 次选：xyxyxyxy（四边形 8 点）
    elif hasattr(result, "obb") and hasattr(result.obb, "xyxyxyxy") and result.obb.xyxyxyxy is not None:
        polys = result.obb.xyxyxyxy.cpu().numpy()     # (N,8)
        conf  = result.obb.conf.cpu().numpy()
        cls   = result.obb.cls.cpu().numpy()
        xywhr = None
    else:
        # 兜底：无 OBB，用 axis-aligned bbox
        if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
            xywh = result.boxes.xywh.cpu().numpy()    # (N,4)
            conf = result.boxes.conf.cpu().numpy()
            cls  = result.boxes.cls.cpu().numpy()
            xywhr = np.concatenate([xywh, np.zeros((xywh.shape[0],1),dtype=np.float32)], axis=1)
            polys = None
        else:
            return None, None, None

    return xywhr, polys, (conf, cls)

def poly_to_xywhr(poly8):
    # poly8: (8,) -> (4,2)
    pts = poly8.reshape(4,2).astype(np.float32)
    rect = cv2.minAreaRect(pts)  # (center(x,y), (w,h), angle_deg) angle in [-90,0)
    (cx,cy), (w,h), ang = rect
    # cv2 角度定义是框长边方向相对水平的旋转；统一转为弧度，逆时针为正
    theta = np.deg2rad(ang)
    # 这里的 theta 方向与模型不完全一致也没关系，因为我们只用于旋正
    return np.array([cx,cy,w,h,theta], dtype=np.float32)

# -------------------------
# 应用仿射变换
# -------------------------
def apply_affine_to_points(M, pts):  # pts: (N,2)
    ones = np.ones((pts.shape[0],1), dtype=np.float32)
    homo = np.concatenate([pts, ones], axis=1)   # (N,3)
    out = (M @ homo.T).T                         # (N,2)
    return out

# -------------------------
# 主流程
# -------------------------
def main():
    model = YOLO(OBB_MODEL, task="obb")

    img_paths = sorted(glob.glob(os.path.join(FULL_IMG_DIR, "*.png")) + 
                       glob.glob(os.path.join(FULL_IMG_DIR, "*.jpg")))

    for img_path in tqdm(img_paths, desc="OBB cropping"):
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(FULL_LBL_DIR, base + ".txt")
        if not os.path.exists(lbl_path):
            continue

        im = cv2.imread(img_path, cv2.IMREAD_COLOR)
        H, W = im.shape[:2]
        cls_gt, (cx_gt,cy_gt,bw_gt,bh_gt), kpts_px = read_full_label(lbl_path, W, H, num_kpts=4)

        # OBB 推理
        res = model(im, verbose=False)[0]
        xywhr, polys, info = extract_obb_xywhr(res)
        if xywhr is None and polys is None:
            # 没检测到，跳过或走兜底（这里直接跳过）
            continue
        conf, cls = info

        # 选一个最佳候选（当前只要 lander 类，且置信度最高）
        idxs = np.where((conf >= MIN_CONF) & (cls == CLASS_ID))[0]
        if len(idxs) == 0:
            # 没有合格候选，跳过或兜底
            continue
        best = idxs[np.argmax(conf[idxs])]

        if xywhr is not None:
            cx, cy, w, h, theta = xywhr[best]
        else:
            cx, cy, w, h, theta = poly_to_xywhr(polys[best])

        # 旋正：绕 (cx,cy) 旋转 -theta
        angle_deg = -float(theta) * 180.0 / math.pi
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        rotated = cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)

        # 旋正后的中心
        center_prime = apply_affine_to_points(M, np.array([[cx,cy]], dtype=np.float32))[0]
        cx_p, cy_p = center_prime.tolist()

        # 裁剪框（加 pad）
        w_pad = w * (1.0 + PAD_RATIO*2.0)
        h_pad = h * (1.0 + PAD_RATIO*2.0)
        x0 = int(round(cx_p - w_pad/2))
        y0 = int(round(cy_p - h_pad/2))
        x1 = int(round(cx_p + w_pad/2))
        y1 = int(round(cy_p + h_pad/2))

        # 边界裁剪
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(W, x1), min(H, y1)
        if x1c <= x0c or y1c <= y0c:
            continue

        crop = rotated[y0c:y1c, x0c:x1c].copy()
        Hc, Wc = crop.shape[:2]

        # 真值关键点 -> 旋正 -> 裁剪坐标
        kpts_rot = apply_affine_to_points(M, kpts_px)  # (4,2)
        kpts_crop = kpts_rot.copy()
        kpts_crop[:, 0] -= x0c
        kpts_crop[:, 1] -= y0c

        # 过滤/可见性；这里继续用 (x,y) 2值格式，不写 v
        # 若点落在裁剪外，我们也可以跳过这张，或将点裁剪到边界
        inside = (kpts_crop[:,0] >= 0) & (kpts_crop[:,0] < Wc) & (kpts_crop[:,1] >= 0) & (kpts_crop[:,1] < Hc)
        if not np.all(inside):
            # 简单策略：跳过（也可将越界点标 v=0，如果你切到 3 值格式）
            continue

        # 由关键点反推 bbox（在裁剪坐标）
        x_min, y_min = np.min(kpts_crop[:,0]), np.min(kpts_crop[:,1])
        x_max, y_max = np.max(kpts_crop[:,0]), np.max(kpts_crop[:,1])
        cx_c = (x_min + x_max) / 2.0
        cy_c = (y_min + y_max) / 2.0
        bw_c = (x_max - x_min)
        bh_c = (y_max - y_min)

        # 归一化
        cx_n, cy_n = cx_c / Wc, cy_c / Hc
        bw_n, bh_n = bw_c / Wc, bh_c / Hc
        kpts_n = kpts_crop.copy()
        kpts_n[:,0] /= Wc
        kpts_n[:,1] /= Hc

        # 输出
        out_img = os.path.join(OUT_IMG_DIR, f"{base}.png")
        out_lbl = os.path.join(OUT_LBL_DIR, f"{base}.txt")
        cv2.imwrite(out_img, crop)

        parts = [str(CLASS_ID), f"{cx_n:.6f}", f"{cy_n:.6f}", f"{bw_n:.6f}", f"{bh_n:.6f}"]
        for i in range(4):
            parts.append(f"{kpts_n[i,0]:.6f}")
            parts.append(f"{kpts_n[i,1]:.6f}")
        with open(out_lbl, "w") as f:
            f.write(" ".join(parts))

if __name__ == "__main__":
    main()
