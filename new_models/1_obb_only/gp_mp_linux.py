#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
warnings.filterwarnings("ignore", category=UserWarning, module="gym.logger")

from yolo_state_mp_linux import YoloStateEstimator

import os, time
from datetime import datetime
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
# 使用 envpool 代替 gymnasium
import envpool.classic_control.v2 as classic_control_envpool
from tqdm import tqdm
import matplotlib.pyplot as plt

# ================== GA 可调参数 ==================
POP = 40
KEEP_RATIO = 0.10
N_GEN = 300
EPISODES = 10
HIGH_MUT_RATE = 0.10
LOW_MUT_RATE = 0.02
PENALTY_NO_YOLO = -20.0 # YOLO失效时的惩罚，可以设置得更低，以促使Agent更快结束
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
    """兼容旧版：没有 reset_counters() 就手动清零。"""
    if hasattr(est, "reset_counters"):
        est.reset_counters()
    else:
        est.frames_total = est.frames_lander_ok = est.frames_terrain_ok = 0

def worker_init():
    """每个子进程启动时初始化一次。"""
    torch.set_num_threads(1)  # 避免在子进程里开太多BLAS线程
    # 使用 envpool 代替 gym，并设置 num_envs=1
    env = classic_control_envpool.make("LunarLander-v2-v0", env_type="gym", num_envs=1)
    model = NNPolicy()
    yolo_estimator = YoloStateEstimator(
        lander_model_path="./best_lander_only.pt",
        terrain_model_path="./terrain.pt",
        conf=0.67,
        device="cuda" if torch.cuda.is_available() else "cpu" # 确保使用GPU
    )
    worker_globals['env'] = env
    worker_globals['model'] = model
    worker_globals['yolo_estimator'] = yolo_estimator

def evaluate_ind_worker(vec: np.ndarray):
    """评估单个个体（子进程中运行）。"""
    env = worker_globals['env']
    model = worker_globals['model']
    yolo_estimator = worker_globals['yolo_estimator']

    _safe_reset_counters(yolo_estimator)
    if hasattr(yolo_estimator, "reset_history"):
        yolo_estimator.reset_history()
    set_weights_vector(model, vec)

    total_reward = 0.0
    
    # envpool 需要先 reset
    obs, info = env.reset() 

    for _ in range(EPISODES):
        # envpool 的 reset 和 step 接口与 gym 稍有不同
        done = False
        reward = 0.0
        
        while not done:
            frame = info['render.rgb_array'][0] # 从 info 中获取渲染帧
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 降低YOLO推理频率
            # 僅在 YoloStateEstimator 第一次更新或每 3 步進行一次
            if yolo_estimator.frames_total == 0 or yolo_estimator.frames_total % 3 == 0:
                yolo_obs = yolo_estimator.update(frame)
                if yolo_obs is None:
                    reward += PENALTY_NO_YOLO
                    break
            else:
                # 重用上一次的观测值，只更新速度
                yolo_obs = yolo_estimator.update_velocity_only(frame)
            
            state = torch.tensor(yolo_obs, dtype=torch.float32)
            with torch.no_grad():
                action = int(torch.argmax(model(state)).item())

            obs, r, term, trunc, info = env.step(np.array([action], dtype=np.int32))
            done = term[0] or trunc[0] # envpool 返回批次化的結果
            reward += r[0]

        total_reward += reward

    avg_reward = total_reward / EPISODES
    
    land_ok = getattr(yolo_estimator, 'frames_lander_ok', 0)
    terr_ok = getattr(yolo_estimator, 'frames_terrain_ok', 0)
    tot = getattr(yolo_estimator, 'frames_total', 0)
    p_land = 100 * land_ok / tot if tot > 0 else 0.0
    p_terr = 100 * terr_ok / tot if tot > 0 else 0.0

    return (avg_reward, p_land, p_terr)
# =====================================

def main():
    # ... (main 函数大部分内容保持不变，只需要修改 pool.imap_unordered 的调用方式)
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
                 maxtasksperchild=100) as pool: # 增大 maxtasksperchild 減少進程重啟開銷

        for gen in range(N_GEN):
            gen_start = time.time()
            
            # 使用 map 而非 imap_unordered，以保持順序和進度條
            results = pool.map(evaluate_ind_worker, pop)
            
            fitness = []
            for i, (avg_reward, p_land, p_terr) in enumerate(results):
                fitness.append(avg_reward)
                #print(f"[Gen {gen+1}, Ind {i+1}] Reward: {avg_reward:.1f}, "
                #      f"Lander Acc: {p_land:.1f}%, Terrain Acc: {p_terr:.1f}%")

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

            # 選擇幸存者
            n_surv = max(2, int(POP * KEEP_RATIO))
            surv_idx = np.argsort(f)[-n_surv:]
            surv = [pop[i] for i in surv_idx]
            surv_fit = [f[i] for i in surv_idx]

            # 生成下一代
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
    print(f"訓練完成，總耗時 {elapsed:.1f} 秒。")

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
    print(f"📈 已保存訓練曲線圖：{figfn}")
    plt.show()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()