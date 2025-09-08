# debug_model.py
import cv2
from ultralytics import YOLO

# 1. 加载您的模型
print("[信息] 正在加载模型 lander.pt...")
model = YOLO('lander.pt', task='obb')

# 2. 读取您准备的测试图片
print("[信息] 正在读取图片 00011.png...")
frame = cv2.imread('00011.png')

# 3. 运行一次预测
print("[信息] 正在运行预测...")
results = model(frame, conf=0.70)[0] # 使用和主脚本相同的置信度

# 4. 深入检查结果对象
print("\n--- 模型原始输出结果分析 ---")
print(f"1. 是否有检测框 (results.boxes): {'是' if results.boxes and results.boxes.conf.nelement() > 0 else '否'}")
print(f"2. 是否有旋转框数据 (results.obb): {'是' if results.obb and results.obb.xywhr.nelement() > 0 else '否'}")

# 5. 打印最关键的信息
print("\n--- 详细数据 ---")
print("results.obb 的内容:")
print(results.obb) # 直接打印OBB对象，看看它到底是什么

# 6. 显示结果图像以供核对
print("\n[信息] 按任意键退出...")
rendered_frame = results.plot()
cv2.imshow("Debug Output", rendered_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()