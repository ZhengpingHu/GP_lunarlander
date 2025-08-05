#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
warnings.filterwarnings("ignore", category=UserWarning, module="gym.logger")

from yolo_state_mp_frame_skip import YoloStateEstimator

import os, time
from datetime import datetime
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# ================== GA 可调参数 ==================
POP = 400
KEEP_RATIO = 0.10
N_GEN = 300
EPISODES = 10
HIGH_MUT_RATE = 0.10
LOW_MUT_RATE = 0.02
PENALTY_NO_YOLO = -20.0
YOLO_INFERENCE_FREQ = 5
# =================================================

# ============== NNPolicy 网络结构 ==============
class NNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.netA = nn.Sequential(nn.Linear(2, 32), nn.ReLU())
        self.netB = nn.Sequential(nn.Linear(6, 32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
                                     nn.Linear(64, 4))

    def forward(self, x):
        a, b = x[..., :2], x[..., 2:]
        return self.fusion(torch.cat([self.netA(a), self.netB(b)], dim=-1))
# ===============================================

# ----------------- 工具函数 -----------------
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

# ============= 多进程相关 =============
worker_globals = {}

def _safe_reset_counters(est):
    if hasattr(est, "reset_counters"):
        est.reset_counters()
    else:
        est.frames_total = est.frames_lander_ok = est.frames_terrain_ok = 0

def worker_init():
    torch.set_num_threads(1)
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    model = NNPolicy()
    yolo_estimator = YoloStateEstimator(
        lander_model_path="./best_lander_only.pt",
        terrain_model_path="./terrain.pt",
        conf=0.67,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    worker_globals['env'] = env
    worker_globals['model'] = model
    worker_globals['yolo_estimator'] = yolo_estimator

def evaluate_ind_worker(vec: np.ndarray):
    env = worker_globals['env']
    model = worker_globals['model']
    yolo_estimator = worker_globals['yolo_estimator']

    _safe_reset_counters(yolo_estimator)
    if hasattr(yolo_estimator, "reset_history"):
        yolo_estimator.reset_history()
    set_weights_vector(model, vec)

    total_reward = 0.0
    
    # 定义每N帧进行一次YOLO推理
    #YOLO_INFERENCE_FREQ = 3 # 每3帧进行一次YOLO推理
    frame_count = 0

    for _ in range(EPISODES):
        obs, _ = env.reset()
        done = False
        reward = 0.0
        
        while not done:
            frame = env.render()
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            yolo_obs = None
            if frame_count % YOLO_INFERENCE_FREQ == 0:
                yolo_obs = yolo_estimator.update_full(frame)
            else:
                yolo_obs = yolo_estimator.predict_only()
                
            frame_count += 1

            if yolo_obs is None:
                reward += PENALTY_NO_YOLO
                break
            
            state = torch.tensor(yolo_obs, dtype=torch.float32)
            with torch.no_grad():
                action = int(torch.argmax(model(state)).item())

            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            reward += r

        total_reward += reward
        frame_count = 0 # 每个episode结束重置帧计数

    avg_reward = total_reward / EPISODES

    land_ok = getattr(yolo_estimator, 'frames_lander_ok', 0)
    terr_ok = getattr(yolo_estimator, 'frames_terrain_ok', 0)
    tot = getattr(yolo_estimator, 'frames_total', 0)
    p_land = 100 * land_ok / tot if tot > 0 else 0.0
    p_terr = 100 * terr_ok / tot if tot > 0 else 0.0

    return (avg_reward, p_land, p_terr)
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

    with mp.Pool(processes=num_processes,
                 initializer=worker_init,
                 maxtasksperchild=100) as pool:

        for gen in range(N_GEN):
            gen_start = time.time()
            results = []
            
            pbar = tqdm(total=len(pop), desc=f"Generation {gen+1}/{N_GEN} - Evaluating")
            for res in pool.imap_unordered(evaluate_ind_worker, pop):
                results.append(res)
                pbar.update(1)
            pbar.close()

            fitness = []
            for i, (avg_reward, p_land, p_terr) in enumerate(results):
                fitness.append(avg_reward)

            f = np.array(fitness, dtype=np.float32)
            avg_f, best_f, worst_f = f.mean(), f.max(), f.min()
            rewards_log['best'].append(best_f)
            rewards_log['mean'].append(avg_f)
            rewards_log['worst'].append(worst_f)

            if best_f > best_global_reward:
                best_global_reward = best_f
                best_global_vec = pop[int(np.argmax(f))].copy()

            print(f"--- Gen {gen+1} Summary ---")
            print(f"Time: {time.time() - gen_start:.2f}s | "
                  f"Best: {best_f:.2f} | Avg: {avg_f:.2f} | Worst: {worst_f:.2f}")
            print("-" * 32)

            n_surv = max(2, int(POP * KEEP_RATIO))
            surv_idx = np.argsort(f)[-n_surv:]
            surv = [pop[i] for i in surv_idx]
            surv_fit = [f[i] for i in surv_idx]

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

    if best_global_vec is None:
        best_global_vec = pop[int(np.argmax(f))]

    set_weights_vector(main_model, best_global_vec)
    wfn = f"best_weights_{tag}.pth"
    torch.save(main_model.state_dict(), wfn)
    print(f"✅ 已保存最佳模型权重：{wfn}")

    figfn = f"training_rewards_{tag}.png"
    plt.figure(figsize=(10, 6))
    gens = list(range(1, N_GEN + 1))
    plt.plot(gens, rewards_log['mean'], label='Mean')
    plt.plot(gens, rewards_log['best'], label='Best')
    plt.fill_between(gens, rewards_log['worst'], rewards_log['best'], alpha=0.2)
    plt.xlabel('Generation')
    plt.ylabel('Reward')
    plt.title('GA Training Rewards')
    plt.legend()
    plt.grid(True)
    plt.savefig(figfn, dpi=150, bbox_inches='tight')
    print(f"📈 已保存训练曲线图：{figfn}")
    plt.show()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()