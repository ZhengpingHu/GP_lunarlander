#!/usr/bin/env python3
# quick_check_and_vis.py
import os, random, cv2, glob
import numpy as np

IMDIR = r"crops_from_obb/images/train"
LBDIR = r"crops_from_obb/labels/train"
OUT   = r"crops_from_obb/_sample_vis"
os.makedirs(OUT, exist_ok=True)

imgs = sorted(glob.glob(os.path.join(IMDIR, "*.png")) + glob.glob(os.path.join(IMDIR, "*.jpg")))
missing = 0
pairs = []
for p in imgs:
    base = os.path.splitext(os.path.basename(p))[0]
    lb = os.path.join(LBDIR, base + ".txt")
    if not os.path.exists(lb):
        missing += 1
    else:
        pairs.append((p, lb))
print(f"Total images: {len(imgs)}, paired: {len(pairs)}, missing label: {missing}")

# 抽查可视化 20 张
sample = random.sample(pairs, min(20, len(pairs)))
for ip, lp in sample:
    im = cv2.imread(ip)
    H, W = im.shape[:2]
    toks = open(lp).read().strip().split()
    arr = list(map(float, toks))
    # 取4个关键点（两值格式）
    k = np.array(arr[5:5+8], dtype=np.float32).reshape(4,2)
    k[:,0] *= W
    k[:,1] *= H

    # 顺序: [左上, 右上, 左脚尖, 右脚尖]
    colors = [(0,0,255),(0,255,255),(0,255,0),(255,0,0)]
    for i,(x,y) in enumerate(k.astype(int)):
        cv2.circle(im, (x,y), 3, colors[i], -1)
        cv2.putText(im, str(i), (x+4,y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[i], 1, cv2.LINE_AA)

    # 脚尖连线（看角度是否合理）
    p2 = k[2].astype(int); p3 = k[3].astype(int)
    cv2.line(im, tuple(p2), tuple(p3), (255,255,255), 1, cv2.LINE_AA)

    cv2.imwrite(os.path.join(OUT, os.path.basename(ip)), im)

print(f"Saved samples to: {OUT}")
