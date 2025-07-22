#!/usr/bin/env python3
import argparse, time, subprocess, sys, os
import numpy as np
import cv2
from ultralytics import YOLO
import gymnasium as gym

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lander-model',  required=True)
    p.add_argument('--terrain-model', required=True)
    p.add_argument('--conf', type=float, default=0.67)
    p.add_argument('--fps', type=float, default=60.0)
    args = p.parse_args()

    model_lander = YOLO(args.lander_model, task="obb")
    model_terrain = YOLO(args.terrain_model, task="pose")

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset()
    interval = 1.0 / args.fps

    while True:
        frame = env.render()
        results_l = model_lander.predict(source=frame, imgsz=frame.shape[:2][::-1],
                                         conf=args.conf, verbose=False)[0]
        frame = results_l.plot(img=frame)
        results_t = model_terrain.predict(source=frame, imgsz=frame.shape[:2][::-1],
                                          conf=args.conf, verbose=False)[0]
        frame = results_t.plot(img=frame)

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("YOLO Pose+OBB", cv2.resize(bgr, (600,400)))

        if cv2.waitKey(int(interval*1000)) & 0xFF in (27, ord('q')):
            break

        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            obs, _ = env.reset()

    cv2.destroyAllWindows()
    env.close()

if __name__ == '__main__':
    main()
