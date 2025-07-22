#!/usr/bin/env python3
# env.py

import gymnasium as gym
import time

from gymnasium.envs.registration import register

import fixed_env

register(
    id="FixedLander-v3",
    entry_point="fixed_env:FixedLander",
)

def main():
    env = gym.make("LunarLander-v3", render_mode="rgb_array", gravity=-3.5)
    obs, info = env.reset()
    try:
        while True:
            # action = 0
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(0)
            # frame = info['rgb_array']
            # print("obs:", obs[6], obs[7], "reward:", reward)
            time.sleep(0.02)
            if terminated or truncated:
                obs, info = env.reset()
                

    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    main()
