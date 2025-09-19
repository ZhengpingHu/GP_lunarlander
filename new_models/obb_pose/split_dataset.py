#!/usr/bin/env python3
# split_dataset.py
import os, glob, random, shutil
random.seed(42)

ROOT = r"crops_from_obb"
IM_TRAIN = os.path.join(ROOT, "images", "train")
LB_TRAIN = os.path.join(ROOT, "labels", "train")

IM_VAL = os.path.join(ROOT, "images", "val");   os.makedirs(IM_VAL, exist_ok=True)
LB_VAL = os.path.join(ROOT, "labels", "val");   os.makedirs(LB_VAL, exist_ok=True)
IM_TEST= os.path.join(ROOT, "images", "test");  os.makedirs(IM_TEST, exist_ok=True)
LB_TEST= os.path.join(ROOT, "labels", "test");  os.makedirs(LB_TEST, exist_ok=True)

imgs = sorted(glob.glob(os.path.join(IM_TRAIN, "*.png")) + glob.glob(os.path.join(IM_TRAIN, "*.jpg")))
n = len(imgs)
n_val  = n // 10
n_test = n // 10

sample = imgs.copy()
random.shuffle(sample)
val_set  = set(sample[:n_val])
test_set = set(sample[n_val:n_val+n_test])

def move_pair(img_path, dst_img_dir, dst_lbl_dir):
    base = os.path.splitext(os.path.basename(img_path))[0]
    ext  = os.path.splitext(img_path)[1]
    lbl  = os.path.join(LB_TRAIN, base + ".txt")
    if not os.path.exists(lbl):
        return
    shutil.move(img_path, os.path.join(dst_img_dir, base + ext))
    shutil.move(lbl,     os.path.join(dst_lbl_dir, base + ".txt"))

for p in imgs:
    if p in val_set:
        move_pair(p, IM_VAL, LB_VAL)
    elif p in test_set:
        move_pair(p, IM_TEST, LB_TEST)
# 其余留在 train
print("Done splitting.")
