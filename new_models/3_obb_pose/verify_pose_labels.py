#!/usr/bin/env python3
"""
verify_pose_labels.py

A utility script to visualize the generated YOLO Pose labels on the images
to ensure the data generation process was accurate.

(一个用于将生成的YOLO Pose标签可视化地绘制在图片上的工具脚本，
以确保数据生成过程的准确性。)
"""

import argparse
import os
import glob
import cv2

def main():
    p = argparse.ArgumentParser(description="YOLO Pose Label Verification Tool")
    p.add_argument('--path', default='training_data_pose', 
                       help="Path to the root directory of the pose dataset")
    args = p.parse_args()

    label_dir = os.path.join(args.path, 'labels')
    image_dir = os.path.join(args.path, 'images')

    if not os.path.exists(label_dir) or not os.path.exists(image_dir):
        print(f"[错误] 在 '{args.path}' 中找不到 'images' 或 'labels' 文件夹。")
        return

    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    
    if not label_files:
        print(f"[错误] 在 '{label_dir}' 中没有找到任何标签文件。")
        return

    print(f"[信息] 找到了 {len(label_files)} 个标签文件。")
    print("[信息] 按 'q' 键退出，按任意其他键查看下一张图片。")

    for label_file in label_files:
        base_filename = os.path.splitext(os.path.basename(label_file))[0]
        image_file = os.path.join(image_dir, f"{base_filename}.png")

        if not os.path.exists(image_file):
            print(f"[警告] 找不到对应的图片: {image_file}")
            continue

        image = cv2.imread(image_file)
        h, w, _ = image.shape

        with open(label_file, 'r') as f:
            line = f.readline().strip()
            parts = [float(x) for x in line.split(' ')]

            # 1. 解析BBox并绘制
            class_id, x_center, y_center, width, height = parts[:5]
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2) # 蓝色BBox

            # 2. 解析关键点并绘制
            keypoints = parts[5:]
            for i in range(0, len(keypoints), 2):
                kpt_x = int(keypoints[i] * w)
                kpt_y = int(keypoints[i+1] * h)
                cv2.circle(image, (kpt_x, kpt_y), 3, (0, 255, 0), -1) # 绿色关键点
                
                # 标注关键点索引
                kpt_index = i // 2
                cv2.putText(image, str(kpt_index), (kpt_x + 5, kpt_y + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Label Verification", image)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("[信息] 验证完成。")

if __name__ == '__main__':
    main()