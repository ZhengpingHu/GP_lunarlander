#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
warnings.filterwarnings("ignore", category=UserWarning, module="gym.logger")

# from yolo_state_mp_frame_skip import YoloStateEstimator # <--- 已移除YOLO依赖

import os, time
from datetime import datetime
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
# import cv2 # <--- 不再需要
from tqdm import tqdm
import matplotlib.pyplot as plt

# ================== GA 可调参数 ==================
POP = 400
KEEP_RATIO = 0.10
N_GEN = 300
EPISODES = 10
HIGH_MUT_RATE = 0.10
LOW_MUT_RATE = 0.02
# PENALTY_NO_YOLO = -20.0 # <--- 不再需要
# YOLO_INFERENCE_FREQ = 5 # <--- 不再需要

# ================== 新增：噪声等级 ==================
# 您可以调整这个值来控制添加到环境观测值中的噪声强度
# 0.0 表示没有噪声
NOISE_LEVEL = 0.1
# =================================================

# ============== NNPolicy 网络结构 (保持不变) ==============
class NNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # LunarLander-v3环境的状态是8维，正好匹配网络输入
        # 前2个是位置(x, y)，后6个是其他状态
        self.netA = nn.Sequential(nn.Linear(2, 32), nn.ReLU())
        self.netB = nn.Sequential(nn.Linear(6, 32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
                                      nn.Linear(64, 4))

    def forward(self, x):
        a, b = x[..., :2], x[..., 2:]
        return self.fusion(torch.cat([self.netA(a), self.netB(b)], dim=-1))
# ===============================================

# ----------------- 工具函数 (保持不变) -----------------
def get_weights_vector(m: nn.Module) -> np.ndarray:
    return torch.cat([p.data.flatten() for p in m.parameters()]).cpu().numpy()

def set_weights_vector(m: nn.Module, vec: np.ndarray):
    ptr = 0
    for p in m.parameters():
        num = p.numel()
        p.data.copy_(torch.from_numpy(vec[ptr:ptr + num]).view_as(p))
        ptr += num

def uniform_crossover(p1, p2):
    mask = np.random.rand(len(p1)) < 0.5
    return np.where(mask, p1, p2), np.where(~mask, p1, p2)

def mutate(vec, rate):
    return vec + np.random.randn(len(vec)) * rate
# -------------------------------------------

# ============= 多进程相关 (已修改) =============
worker_globals = {}

def worker_init():
    """初始化每个工作进程。不再需要YOLO。"""
    torch.set_num_threads(1)
    # render_mode可以设为None，因为我们不再需要渲染图像
    env = gym.make("LunarLander-v3") 
    model = NNPolicy()
    
    worker_globals['env'] = env
    worker_globals['model'] = model

def evaluate_ind_worker(vec: np.ndarray):
    """
    修改后的评估函数。
    输入源现在是环境的原始状态 + 随机噪声。
    """
    env = worker_globals['env']
    model = worker_globals['model']

    set_weights_vector(model, vec)

    total_reward = 0.0
    
    for _ in range(EPISODES):
        obs, _ = env.reset()
        done = False
        reward = 0.0
        
        while not done:
            # 1. 给环境观测值添加噪声
            noisy_obs = obs + np.random.randn(*obs.shape) * NOISE_LEVEL
            
            # 2. 将带噪声的观测值作为模型输入
            state = torch.tensor(noisy_obs, dtype=torch.float32)
            with torch.no_grad():
                action = int(torch.argmax(model(state)).item())

            # 3. 执行动作并获取新的状态
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            reward += r

        total_reward += reward
        
    avg_reward = total_reward / EPISODES

    # 4. 返回值简化，只返回平均奖励
    return avg_reward
# =====================================

def main():
    main_model = NNPolicy()
    dim = sum(p.numel() for p in main_model.parameters())

    pop = [np.random.randn(dim) for _ in range(POP)]
    rewards_log = {'best': [], 'mean': [], 'worst': []}
    best_global_reward = -np.inf
    best_global_vec = None

    start_time = time.time()
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    num_processes = max(1, min(POP, os.cpu_count() or 1) - 2)
    print(f"🚀 [INFO] Starting GA training with {num_processes} persistent parallel processes.")
    print(f"🔬 [INFO] Noise level set to: {NOISE_LEVEL}")

    with mp.Pool(processes=num_processes,
                 initializer=worker_init,
                 maxtasksperchild=100) as pool:

        for gen in range(N_GEN):
            gen_start = time.time()
            
            # 使用 imap_unordered 来获取结果
            # results 将是一个包含 avg_reward 的列表
            results = list(tqdm(pool.imap_unordered(evaluate_ind_worker, pop), 
                                total=len(pop), 
                                desc=f"Generation {gen+1}/{N_GEN} - Evaluating"))

            # 直接使用results作为fitness列表
            fitness = np.array(results, dtype=np.float32)
            
            avg_f, best_f, worst_f = fitness.mean(), fitness.max(), fitness.min()
            rewards_log['best'].append(best_f)
            rewards_log['mean'].append(avg_f)
            rewards_log['worst'].append(worst_f)

            if best_f > best_global_reward:
                best_global_reward = best_f
                best_global_vec = pop[int(np.argmax(fitness))].copy()

            print(f"--- Gen {gen+1} Summary ---")
            print(f"Time: {time.time() - gen_start:.2f}s | "
                  f"Best: {best_f:.2f} | Avg: {avg_f:.2f} | Worst: {worst_f:.2f}")
            print("-" * 32)

            n_surv = max(2, int(POP * KEEP_RATIO))
            surv_idx = np.argsort(fitness)[-n_surv:]
            surv = [pop[i] for i in surv_idx]
            surv_fit = [fitness[i] for i in surv_idx]

            new_pop = surv.copy()
            while len(new_pop) < POP:
                pa, pb = np.random.choice(len(surv), 2, replace=False)
                c1, c2 = uniform_crossover(surv[pa], surv[pb])

                rate1 = HIGH_MUT_RATE if surv_fit[pa] < avg_f else LOW_MUT_RATE
                rate2 = HIGH_MUT_RATE if surv_fit[pb] < avg_f else LOW_MUT_RATE

                new_pop.append(mutate(c1, rate1))
                if len(new_pop) < POP:
                    new_pop.append(mutate(c2, rate2))

            pop = new_pop

    elapsed = time.time() - start_time
    print(f"训练完成，总耗时 {elapsed:.1f} 秒。")

    if best_global_vec is None and len(fitness) > 0:
        best_global_vec = pop[int(np.argmax(fitness))]

    set_weights_vector(main_model, best_global_vec)
    wfn = f"best_weights_{tag}_noise{NOISE_LEVEL}.pth"
    torch.save(main_model.state_dict(), wfn)
    print(f"✅ 已保存最佳模型权重：{wfn}")

    figfn = f"training_rewards_{tag}_noise{NOISE_LEVEL}.png"
    plt.figure(figsize=(10, 6))
    gens = list(range(1, N_GEN + 1))
    plt.plot(gens, rewards_log['mean'], label='Mean')
    plt.plot(gens, rewards_log['best'], label='Best')
    plt.fill_between(gens, rewards_log['worst'], rewards_log['best'], alpha=0.2)
    plt.xlabel('Generation')
    plt.ylabel('Reward')
    plt.title(f'GA Training Rewards (Noise Level: {NOISE_LEVEL})')
    plt.legend()
    plt.grid(True)
    plt.savefig(figfn, dpi=150, bbox_inches='tight')
    print(f"📈 已保存训练曲线图：{figfn}")
    # plt.show() # 在服务器运行时可以注释掉


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()