#!/usr/bin/env python3
"""
generate_pose_data.py

This script is dedicated to creating a high-quality dataset for training the 
Stage-2 YOLO Pose model. It uses a robust mathematical transformation to 
generate perfectly accurate keypoint labels from the environment's physics state.
It also includes a powerful '--debug-draw' mode to visualize the underlying physics.

(该脚本专门用于为第二阶段的YOLO Pose模型创建高质量数据集。
它使用一个稳健的数学转换，从环境的物理状态中生成绝对精确的关键点标签。
同时包含一个强大的'--debug-draw'模式，用于可视化底层物理信息。)
"""

import argparse
import os
import numpy as np
import cv2
import gymnasium as gym

# --- 从环境中导入我们需要的物理和尺寸常量 ---
from gymnasium.envs.box2d.lunar_lander import (
    SCALE,
    LANDER_POLY,
    LEG_H,
    VIEWPORT_W,
    VIEWPORT_H,
)


def create_output_directory(path):
    """如果输出目录不存在，则创建它。"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[信息] 已创建输出目录: {path}")


def world_to_screen_coords(pos_world, width=VIEWPORT_W, height=VIEWPORT_H, scale=SCALE):
    """
    将 Box2D 世界坐标转换为屏幕像素坐标
    （与 LunarLander 的官方渲染逻辑保持一致）
    """
    screen_x = pos_world.x * scale
    screen_y = height - pos_world.y * scale
    return (screen_x, screen_y)


def get_keypoints_from_env(env):
    """
    深入环境内部，获取所有关键点的精确全局像素坐标。
    """
    lander = env.unwrapped.lander
    legs = env.unwrapped.legs

    # 1. 获取机身左上角和右上角 (keypoints 0, 1)
    top_left_local = LANDER_POLY[0]
    top_right_local = LANDER_POLY[5]
    
    top_left_world = lander.GetWorldPoint((top_left_local[0]/SCALE, top_left_local[1]/SCALE))
    top_right_world = lander.GetWorldPoint((top_right_local[0]/SCALE, top_right_local[1]/SCALE))

    top_left_screen = world_to_screen_coords(top_left_world)
    top_right_screen = world_to_screen_coords(top_right_world)
    
    # 2. 获取左脚尖和右脚尖 (keypoints 2, 3)
    foot_tips_screen = []
    for leg_obj in legs:
        foot_tip_local = (0, -LEG_H / SCALE / 2)
        foot_tip_world = leg_obj.GetWorldPoint(foot_tip_local)
        foot_tip_screen = world_to_screen_coords(foot_tip_world)
        foot_tips_screen.append(foot_tip_screen)

    left_foot_tip, right_foot_tip = foot_tips_screen[0], foot_tips_screen[1]

    # 按照 [左上角, 右上角, 左脚尖, 右脚尖] 的顺序返回
    return [top_left_screen, top_right_screen, left_foot_tip, right_foot_tip]


def draw_physics_bodies(overlay, env):
    """
    在给定的图像上，直接绘制出物理引擎中的形状。
    """
    lander = env.unwrapped.lander
    legs = env.unwrapped.legs

    # 绘制机身
    lander_shape = lander.fixtures[0].shape
    vertices = [lander.transform * v for v in lander_shape.vertices]
    screen_vertices = [world_to_screen_coords(v) for v in vertices]
    cv2.polylines(overlay, [np.array(screen_vertices, np.int32)], True, (255, 0, 0, 128), 2)

    # 绘制腿
    for leg in legs:
        leg_shape = leg.fixtures[0].shape
        vertices = [leg.transform * v for v in leg_shape.vertices]
        screen_vertices = [world_to_screen_coords(v) for v in vertices]
        cv2.polylines(overlay, [np.array(screen_vertices, np.int32)], True, (0, 255, 0, 128), 2)


def main():
    p = argparse.ArgumentParser(description="YOLO Pose Dataset Generator for LunarLander")
    p.add_argument('--num-frames', type=int, default=100, help="Number of frames/labels to generate")
    p.add_argument('--debug-draw', action='store_true', help="Draw physics bodies directly on the image for debugging")
    args = p.parse_args()

    img_path = "training_data_pose/images"
    lbl_path = "training_data_pose/labels"
    create_output_directory(img_path)
    create_output_directory(lbl_path)

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset()

    print(f"[信息] 将生成 {args.num_frames} 组图片和Pose标签...")

    frame_count = 0
    while frame_count < args.num_frames:
        frame = env.render()
        
        keypoints = get_keypoints_from_env(env)
        
        yolo_kpts = []
        valid_frame = True
        for (x, y) in keypoints:
            if not (0 < x < VIEWPORT_W and 0 < y < VIEWPORT_H):
                valid_frame = False
                break
            yolo_kpts.extend([x / VIEWPORT_W, y / VIEWPORT_H])

        if valid_frame:
            all_x = [kp[0] for kp in keypoints]
            all_y = [kp[1] for kp in keypoints]
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            
            class_id = 0
            bbox_center_x_norm = ((x_min + x_max) / 2) / VIEWPORT_W
            bbox_center_y_norm = ((y_min + y_max) / 2) / VIEWPORT_H
            bbox_width_norm = (x_max - x_min) / VIEWPORT_W
            bbox_height_norm = (y_max - y_min) / VIEWPORT_H

            base_filename = f"frame_{frame_count:05d}"
            img_filepath = os.path.join(img_path, f"{base_filename}.png")
            
            if args.debug_draw:
                overlay = np.zeros((VIEWPORT_H, VIEWPORT_W, 4), dtype=np.uint8)
                draw_physics_bodies(overlay, env)
                frame_bgra = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
                alpha = overlay[:, :, 3] / 255.0
                for c in range(0, 3):
                    frame_bgra[:, :, c] = frame_bgra[:, :, c] * (1 - alpha) + overlay[:, :, c] * alpha
                cv2.imwrite(img_filepath, frame_bgra)
            else:
                cv2.imwrite(img_filepath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            lbl_filepath = os.path.join(lbl_path, f"{base_filename}.txt")
            
            label_parts = [str(class_id), 
                           f"{bbox_center_x_norm:.6f}", f"{bbox_center_y_norm:.6f}",
                           f"{bbox_width_norm:.6f}", f"{bbox_height_norm:.6f}"]
            for i in range(len(yolo_kpts)):
                label_parts.append(f"{yolo_kpts[i]:.6f}")
                
            yolo_format_string = " ".join(label_parts)
            
            with open(lbl_filepath, 'w') as f:
                f.write(yolo_format_string)
            
            frame_count += 1

        progress = frame_count / args.num_frames
        print(f"\r[信息] 正在处理第 {frame_count} 组数据... 进度: [{'#' * int(progress * 20):<20}] {progress:.0%}", end="")
        
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
            
    print(f"\n[信息] Pose数据集生成完毕。")
    env.close()


if __name__ == '__main__':
    main()
