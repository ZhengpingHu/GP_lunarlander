# save as quick_bias_find.py and run:  python quick_bias_find.py analysis_results_fused/fused_report.csv
import sys, math, pandas as pd, numpy as np

def wrap_pi(a):
    while a <= -math.pi: a += 2*math.pi
    while a > math.pi: a -= 2*math.pi
    return a

def ang_diff(a, b):  # radians
    return abs(wrap_pi(a-b))

csv_path = sys.argv[1] if len(sys.argv)>1 else "analysis_results_fused/fused_report.csv"
df = pd.read_csv(csv_path)

# 取存在 OBB 预测和真值的样本
mask = df["obb_deg"].notna() & df["true_deg"].notna()
obb = np.deg2rad(df.loc[mask, "obb_deg"].values.astype(float))
tru = np.deg2rad(df.loc[mask, "true_deg"].values.astype(float))

best_bias = 0.0
best_mae  = 1e9
for bias_deg in range(-135, 136, 5):  # 粗扫，用5°步长足够
    bias = math.radians(bias_deg)
    mae = np.mean([abs(ang_diff(o+bias, t)) for o, t in zip(obb, tru)])
    if mae < best_mae:
        best_mae = mae
        best_bias = bias_deg

print(f"Best OBB bias ≈ {best_bias:+d} deg (OBB MAE ~ {math.degrees(best_mae):.2f}°)")
