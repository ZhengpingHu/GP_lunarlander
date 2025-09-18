#!/usr/bin/env python3
"""
run_pipeline.py

This script implements the full two-stage detection pipeline for the LunarLander.
It serves as the final application, using the trained models to extract
structured data from the game in real-time.

(该脚本实现了完整的、用于月球着陆器的两阶段识别流水线。
它作为最终的应用程序，使用我们训练好的模型，从游戏中实时提取结构化数据。)
"""

import argparse
import sys
import os
import math
import numpy as np
import cv2
from ultralytics import YOLO
import gymnasium as gym

# --- 导入我们需要的物理和尺寸常量 ---
from gymnasium.envs.box2d.lunar_lander import VIEWPORT_W, VIEWPORT_H

def main():
    p = argparse.ArgumentParser(description="Two-Stage Lander Detection Pipeline")
    p.add_argument('--obb-model', required=True, help="Path to the trained Stage-1 OBB model")
    p.add_argument('--pose-model', required=True, help="Path to the trained Stage-2 Pose model")
    p.add_argument('--conf', type=float, default=0.75, help="Confidence threshold for the initial OBB detection")
    args = p.parse_args()

    # --- 1. 加载我们的“专家团队” ---
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
    
    print("[信息] 流水线启动。按 'q' 键退出。")

    while True:
        # --- 获取当前游戏画面 ---
        original_frame = env.render()
        
        # 创建一个副本用于绘制最终结果，保持原始帧干净
        display_frame = original_frame.copy()

        # --- 阶段一: 全局定位 (OBB Model) ---
        obb_results = obb_model(original_frame, conf=args.conf, verbose=False)[0]

        # 检查OBB模型是否成功找到了着陆器
        if obb_results.obb and obb_results.obb.xywhr.nelement() > 0:
            
            # --- 阶段二: 局部裁剪 (The "Cut") ---
            
            # 1. 获取能包围旋转框的最小正立矩形
            # obb_results.obb.xyxy[0] 提供了 [x_min, y_min, x_max, y_max]
            box = obb_results.obb.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            
            # 2. 增加安全边距 (Padding)
            padding = 20 # 在四周各增加20个像素
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(VIEWPORT_W, x2 + padding)
            y2 = min(VIEWPORT_H, y2 + padding)

            # 3. 执行裁剪
            cropped_image = original_frame[y1:y2, x1:x2]

            # --- 阶段三: 精细识别 (Pose Model) ---
            # 我们将裁剪出的小图送入Pose模型。
            # YOLO库会自动处理缩放、填充至标准尺寸（如320x320）的复杂工作。
            pose_results = pose_model(cropped_image, verbose=False)[0]
            
            # --- 阶段四: 坐标还原 (Mapping Back) ---
            if pose_results.keypoints and pose_results.keypoints.xy.nelement() > 0:
                
                # pose_results.keypoints.xy 会返回在 *裁剪图* 中的坐标
                local_keypoints = pose_results.keypoints.xy[0].cpu().numpy().astype(int)
                
                for i, (local_x, local_y) in enumerate(local_keypoints):
                    # 进行简单的加法，将局部坐标还原为全局坐标
                    global_x = x1 + local_x
                    global_y = y1 + local_y
                    
                    # 在最终的显示画面上，绘制出我们精确找到的关键点
                    cv2.circle(display_frame, (global_x, global_y), 3, (0, 255, 0), -1)
                    cv2.putText(display_frame, str(i), (global_x + 5, global_y + 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # --- 显示最终结果 ---
        cv2.imshow("Two-Stage Pipeline Output", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
            
    env.close()
    cv2.destroyAllWindows()
    print("[信息] 流水线已关闭。")

if __name__ == '__main__':
    main()