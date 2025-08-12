#!/usr/bin/env python3
"""
env_yolo_live.py
用于在随机 LunarLander-v3 环境中实时测试两个 YOLOv11-Pose 模型（lander + terrain）
"""

import argparse, time, subprocess, sys, os
import numpy as np
import cv2
from ultralytics import YOLO
import gymnasium as gym

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lander-model',  required=True, help="YOLOv11-Pose model for lander")
    p.add_argument('--terrain-model', required=True, help="YOLOv11-Pose model for terrain")
    p.add_argument('--conf',          type=float, default=0.5, help="confidence threshold")
    p.add_argument('--fps',           type=float, default=60.0, help="max FPS")
    p.add_argument('--env-script',    default='env.py', help="path to env.py script")
    args = p.parse_args()

    # 1) 启动环境脚本
    python_exec = sys.executable
    if not os.path.isfile(args.env_script):
        print(f"[ERROR] cannot find {args.env_script}")
        sys.exit(1)
    proc = subprocess.Popen([python_exec, args.env_script],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    print(f"[INFO] launch '{args.env_script}' (pid={proc.pid})，waiting for env to stabilize…")
    time.sleep(3)

    # 2) 加载 YOLO-Pose 模型
    print(f"[INFO] load lander model {args.lander_model}")
    model_lander = YOLO(args.lander_model, task="pose")
    print(f"[INFO] load terrain model {args.terrain_model}")
    model_terrain = YOLO(args.terrain_model, task="pose")

    # 3) 创建环境
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset()
    interval = 1.0 / args.fps

    try:
        while True:
            frame = env.render()
            rendered = frame.copy()

            # YOLO-Pose 推理与叠加绘制
            results_lander = model_lander(frame, conf=args.conf)[0]
            rendered = results_lander.plot(img=rendered, conf=False)

            results_terrain = model_terrain(frame, conf=args.conf)[0]
            rendered = results_terrain.plot(img=rendered, conf=False)

            # 显示窗口
            bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            cv2.imshow("YOLOv11-Pose: Lander + Terrain", cv2.resize(bgr, (600, 400)))

            key = cv2.waitKey(int(interval * 1000)) & 0xFF
            if key in (27, ord('q')):
                break

            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

    except Exception as e:
        print(f"[ERROR] Runtime exception: {e}")

    finally:
        cv2.destroyAllWindows()
        env.close()
        proc.terminate()
        print(f"[INFO] end script pid={proc.pid}")

if __name__ == '__main__':
    main()
