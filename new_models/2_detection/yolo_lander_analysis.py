#!/usr/bin/env python3
"""
yolo_lander_analysis.py (OBB Version)

This script analyzes the performance of a YOLOv8-OBB model for the LunarLander environment.
It directly extracts the angle from the Oriented Bounding Box (OBB) results.

(该脚本用于分析YOLOv8-OBB模型在月球着陆器环境中的性能。
它会直接从定向边界框(OBB)结果中提取角度。)
"""

import argparse
import sys
import os
import math
import numpy as np
import cv2
from ultralytics import YOLO
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd

# --- 导入您的自定义环境文件 ---
# 注意: 即使在这个脚本中不直接使用fixed_env模式，
# 导入它也是必要的，因为它会在Gymnasium中注册'FixedLander-v0'环境。
import fixed_env

# --- 辅助函数 ---

def get_angle_from_obb(obb_results):
    """
    从YOLO OBB结果中计算着陆器的角度（单位：度）。
    """
    try:
        if obb_results and obb_results.xywhr.nelement() > 0:
            angle_rad_yolo = obb_results.xywhr[0][4].item()
            angle_deg_yolo = math.degrees(angle_rad_yolo)
            
            # --- 角度校准 ---
            predicted_angle = angle_deg_yolo + 90
            
            while predicted_angle > 180: predicted_angle -= 360
            while predicted_angle < -180: predicted_angle += 360
            
            return predicted_angle
        return None
    except (IndexError, TypeError):
        return None

def create_output_directory(path):
    """如果输出目录不存在，则创建它。"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[信息] 已创建输出目录: {path}")

def calculate_angular_error(true_angle, pred_angle):
    """
    计算两个角度之间的最短绝对角误差。
    """
    error = pred_angle - true_angle
    # 将误差归一化到[-180, 180]的范围内
    while error > 180:
        error -= 360
    while error < -180:
        error += 360
    return abs(error)


# --- 分析与绘图 ---

def generate_report(data, output_dir, report_suffix=''):
    """
    分析收集到的数据并生成图表和摘要报告。
    """
    if not data:
        print(f"\n[警告] {report_suffix.replace('_','').upper()} 模式未收集到任何有效数据，无法生成报告。")
        print("[提示] 您可以尝试使用 '--visualize' 选项运行，或调整 '--conf' 阈值。")
        return

    mode_name = report_suffix.replace('_','').upper() or "ANALYSIS"
    print(f"\n--- 正在为【{mode_name}】模式生成报告 ---")

    df = pd.DataFrame(data, columns=['true_angle', 'predicted_angle', 'confidence', 'abs_error'])
    
    df['angle_bin'] = df['true_angle'].round().astype(int)
    error_by_angle = df.groupby('angle_bin')['abs_error'].mean()

    plt.figure(figsize=(15, 7))
    error_by_angle.plot(kind='bar', color='skyblue')
    plt.title(f'[{mode_name}] Average Absolute Angle Error vs. True Angle', fontsize=16)
    plt.xlabel('True Angle (Degrees) (真实角度)', fontsize=12)
    plt.ylabel('Average Absolute Error (Degrees) (平均绝对误差)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    ax = plt.gca()
    ticks = ax.get_xticks()
    labels = ax.get_xticklabels()
    n = max(1, len(ticks) // 20)
    ax.set_xticks(ticks[::n])
    ax.set_xticklabels(labels[::n])
    
    plot1_path = os.path.join(output_dir, f'error_vs_angle{report_suffix}.png')
    plt.savefig(plot1_path)
    plt.close()
    print(f"[信息] 已将弱点分析图保存至 {plot1_path}")

    plt.figure(figsize=(12, 7))
    plt.scatter(df['confidence'], df['abs_error'], alpha=0.5, label='Data Point')
    plt.axhline(y=2.0, color='r', linestyle='--', label='2-degree Error Threshold')
    plt.title(f'[{mode_name}] Absolute Error vs. Model Confidence', fontsize=16)
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Absolute Error (Degrees)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plot2_path = os.path.join(output_dir, f'error_vs_confidence{report_suffix}.png')
    plt.savefig(plot2_path)
    plt.close()
    print(f"[信息] 已将置信度分析图保存至 {plot2_path}")

    if not error_by_angle.empty:
        weakest_angle = error_by_angle.idxmax()
        max_error = error_by_angle.max()
        print(f"该模式下模型表现最差的角度约为 {weakest_angle:.1f} 度，平均误差为 {max_error:.2f} 度。")
    
    suggested_conf = 0.70
    for conf_threshold in np.arange(0.70, 1.0, 0.01):
        subset = df[df['confidence'] >= conf_threshold]
        if len(subset) > 10:
            if (subset['abs_error'] < 2.0).mean() > 0.95:
                suggested_conf = conf_threshold
                break
    
    print(f"为确保95%的预测误差低于2度，该模式下建议的置信度阈值为 {suggested_conf:.2f}。")
    print("--- 报告生成完毕 ---")

# --- 主要执行模式 ---

def run_datalogger_analysis(args, model):
    """
    运行一个简单的数据记录会话，持续固定的帧数。
    """
    print("\n--- 开始 'DATALOGGER' 数据记录模式 ---")
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset()
    collected_data = []
    
    print(f"[信息] 将运行 {args.datalogger_frames} 帧并记录所有成功检测...")

    for frame_count in range(1, args.datalogger_frames + 1):
        frame = env.render()
        true_angle_rad = obs[4]
        true_angle_deg = math.degrees(true_angle_rad)

        results = model(frame, conf=args.conf, verbose=False)[0]
        
        debug_msg = ""
        
        # **FIX**: 这是最终的、正确的检测逻辑
        # 我们只关心 results.obb 是否有数据
        if results.obb and results.obb.conf is not None and results.obb.conf.nelement() > 0:
            pred_angle = get_angle_from_obb(results.obb)
            if pred_angle is not None:
                # 成功获取所有需要的信息
                confidence = results.obb.conf[0].item()
                # **修改**: 使用新的、更精确的误差计算函数
                error = calculate_angular_error(true_angle_deg, pred_angle)
                collected_data.append((true_angle_deg, pred_angle, confidence, error))
                debug_msg = f"Success (Conf: {confidence:.2f})"
            else:
                # 这种情况几乎不会发生，但作为保险
                debug_msg = "Fail: OBB data malformed"
        else:
            # 模型没有返回任何OBB结果
            debug_msg = "Fail: No OBB detection"

        progress = frame_count / args.datalogger_frames
        print(f"\r[信息] 进度: [{'#' * int(progress * 20):<20}] {progress:.0%}", end="")

        if args.visualize:
            rendered_frame = results.plot()
            status_text = f"Angle: {true_angle_deg:.1f} | Status: {debug_msg}"
            cv2.putText(rendered_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Debug Visualization", cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR))
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('p'):
                print("\n[调试] 已暂停。在窗口激活时按任意键继续...")
                cv2.waitKey(0)
        
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
            
    print("\n[信息] 'DATALOGGER' 模式数据收集完成。")
    env.close()
    return collected_data


def main():
    p = argparse.ArgumentParser(description="YOLOv8-OBB Lander Performance Analysis Tool")
    p.add_argument('--lander-model', required=True, help="Path to YOLOv8-OBB model for the lander")
    p.add_argument('--conf', type=float, default=0.70, help="Confidence threshold for detection")
    p.add_argument('--output-dir', default='analysis_results', help="Directory to save plots and report")
    p.add_argument('--visualize', action='store_true', help="Display frames with model predictions for debugging")
    p.add_argument('--datalogger-frames', type=int, default=5000, help="Number of frames to run in datalogger mode")
    
    args = p.parse_args()

    create_output_directory(args.output_dir)
    print(f"[信息] 正在从 {args.lander_model} 加载着陆器OBB模型")
    try:
        model_lander = YOLO(args.lander_model, task="obb")
    except Exception as e:
        print(f"[错误] 加载YOLO模型失败: {e}")
        sys.exit(1)
    
    try:
        data = run_datalogger_analysis(args, model_lander)
        generate_report(data, args.output_dir, report_suffix='_datalogger')
    finally:
        cv2.destroyAllWindows()
        print("\n--- 所有分析任务已完成！ ---")

if __name__ == '__main__':
    main()