#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Final Integrated Version: Competitive Fitness + Dual Refresh (Performance + Age)

import os
import argparse
import random
import multiprocessing as mp
from multiprocessing.connection import Client
from typing import Optional, Tuple, Sequence, List
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import cv2
from tqdm import tqdm

# [PLOT] 离屏绘图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Boilerplate for Multiprocessing ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
print = lambda *a, **k: (__import__("builtins").print(*a, **{**k, "flush": True}))

# ----------------------------
# 1. Reproducibility
# ----------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    except Exception:
        pass

# ----------------------------
# 2. Neural Network & GA Operators (无变动)
# ----------------------------
class NNPolicy(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, hidden), nn.ReLU(), nn.Linear(hidden, 4))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def get_weights_vector(m: nn.Module) -> np.ndarray:
    with torch.no_grad():
        return torch.cat([p.data.flatten() for p in m.parameters()]).cpu().numpy()

def set_weights_vector(m: nn.Module, vec: np.ndarray):
    offset = 0
    with torch.no_grad():
        for p in m.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(vec[offset:offset+n]).view_as(p))
            offset += n

def uniform_crossover(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.random.rand(len(p1)) < 0.5
    c1 = np.where(mask, p1, p2)
    c2 = np.where(~mask, p1, p2)
    return c1, c2

def mutate(vec: np.ndarray, sigma: float) -> np.ndarray:
    return vec + np.random.randn(vec.size) * sigma

# ----------------------------
# 3. Seed Management System
# ----------------------------
class SeedAgeScheduler:
    def __init__(self, seed_pool: Sequence[int], rng_seed: int = 0):
        self.seed_pool = list(seed_pool)
        self.ages: List[int] = [0 for _ in self.seed_pool] # 年龄列表，与 seed_pool 一一对应
        self.rng = random.Random(rng_seed)

    @classmethod
    def from_fixed_pool(cls, pool_size: int, base_seed: int, shuffle: bool, rng_seed: int):
        g = np.random.default_rng(base_seed)
        pool = g.integers(low=0, high=2**31-1, size=pool_size, dtype=np.int32).tolist()
        if shuffle:
            rr = random.Random(rng_seed)
            rr.shuffle(pool)
        return cls(pool, rng_seed=rng_seed)

    def select_subset(self, k: int) -> List[int]:
        n = len(self.seed_pool)
        if k >= n: return list(self.seed_pool)
        idx = list(range(n))
        self.rng.shuffle(idx)
        idx.sort(key=lambda i: self.ages[i], reverse=True)
        return [self.seed_pool[i] for i in idx[:k]]

    def update_after_generation(self, chosen_seeds: Sequence[int]):
        chosen = set(chosen_seeds)
        for i, s in enumerate(self.seed_pool):
            self.ages[i] = 0 if s in chosen else self.ages[i] + 1
    
    # [NEW] 辅助函数，根据种子ID获取其年龄
    def get_age_of_seed(self, seed: int) -> int:
        try:
            idx = self.seed_pool.index(seed)
            return self.ages[idx]
        except ValueError:
            return -1 # Seed not in pool

    def state_dict(self) -> dict:
        return {"seed_pool": self.seed_pool, "ages": self.ages}

    def load_state_dict(self, d: dict):
        if d["seed_pool"] != self.seed_pool: raise ValueError("Seed pool mismatch.")
        self.ages = list(d["ages"])

class SeedPortfolioManager:
    def __init__(self, pool_size, base_seed, shuffle, pool_rng_seed, subset_k):
        self.scheduler = SeedAgeScheduler.from_fixed_pool(pool_size, base_seed, shuffle, pool_rng_seed)
        self.master_pool = self.scheduler.seed_pool
        self.active_subset = self.scheduler.select_subset(subset_k)
        print(f"🌱 Initial seed subset: {self.active_subset}")

    def get_active_subset(self) -> List[int]:
        return self.active_subset

    # [MODIFIED] 签名增加了 max_seed_age
    def update_and_refresh(self, results_matrix: np.ndarray, refresh_frac: float, refresh_direction: str, success_threshold: float, max_seed_age: int):
        # 1. 更新所有种子的年龄
        self.scheduler.update_after_generation(self.active_subset)
        
        indices_to_replace = set()

        # 2. [NEW] 触发器1：检查年龄替换
        if max_seed_age > 0:
            age_indices_to_replace = set()
            for i, seed in enumerate(self.active_subset):
                age = self.scheduler.get_age_of_seed(seed)
                if age > max_seed_age:
                    age_indices_to_replace.add(i)
            if age_indices_to_replace:
                print(f"Replacing {len(age_indices_to_replace)} seeds due to max age ({max_seed_age}).")
                indices_to_replace.update(age_indices_to_replace)

        # 3. [MODIFIED] 触发器2：检查性能替换
        num_perf_replace = int(np.floor(len(self.active_subset) * refresh_frac))
        if num_perf_replace > 0 and refresh_direction != 'none':
            success_matrix = results_matrix > success_threshold
            solvers_per_seed = success_matrix.sum(axis=0)
            seed_difficulty_scores = 1.0 / (solvers_per_seed + 1e-9)
            sorted_indices_by_difficulty = np.argsort(seed_difficulty_scores)

            perf_indices_to_replace = set()
            if refresh_direction == 'bottom':
                indices = sorted_indices_by_difficulty[-num_perf_replace:]
                perf_indices_to_replace.update(indices)
                print(f"Replacing {len(perf_indices_to_replace)} hardest performing seeds (highest weights).")
            elif refresh_direction == 'top':
                indices = sorted_indices_by_difficulty[:num_perf_replace]
                perf_indices_to_replace.update(indices)
                print(f"Replacing {len(perf_indices_to_replace)} easiest performing seeds (lowest weights).")
            
            indices_to_replace.update(perf_indices_to_replace)

        # 4. 如果没有需要替换的，提前返回
        if not indices_to_replace:
            return

        # 5. 执行替换
        num_to_replace_total = len(indices_to_replace)
        old_seeds_to_replace = {self.active_subset[i] for i in indices_to_replace}
        candidate_pool = [s for s in self.master_pool if s not in self.active_subset]
        
        if len(candidate_pool) < num_to_replace_total:
            print(f"[WARN] Not enough ({len(candidate_pool)}) unused seeds to replace {num_to_replace_total}.")
            # 可用种子不足，能换多少换多少
            num_to_replace_total = len(candidate_pool)
            indices_to_replace = list(indices_to_replace)[:num_to_replace_total]

        if num_to_replace_total == 0:
             print("[WARN] No seeds to replace.")
             return
             
        new_seeds = random.sample(candidate_pool, num_to_replace_total)
        
        for i_replace, i_new in zip(indices_to_replace, range(num_to_replace_total)):
            self.active_subset[i_replace] = new_seeds[i_new]
        
        print(f"  - Replaced: {list(old_seeds_to_replace)}")
        print(f"  + New active subset: {self.active_subset}")
    
    def state_dict(self): return self.scheduler.state_dict()
    def load_state_dict(self, d): self.scheduler.load_state_dict(d)

# ----------------------------
# 4. RPC Client (无变动)
# ----------------------------
class RPCClient:
    def __init__(self, host, port, authkey="yolo-rpc"):
        self.address = (host, port)
        self.authkey = authkey.encode("utf-8")
        self.conn: Optional[Client] = None
    def __enter__(self):
        self.conn = Client(self.address, authkey=self.authkey); return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn: self.conn.close()
    def reset(self):
        self.conn.send(("reset", None)); _ = self.conn.recv()
    def update_full(self, frame_bgr, steps=0):
        self.conn.send(("update_full", (frame_bgr, steps))); ok, z = self.conn.recv(); return z if ok else None
    def predict_only(self):
        self.conn.send(("predict_only", None)); ok, z = self.conn.recv(); return z if ok else None

# ----------------------------
# 5. Evaluation Function (无变动)
# ----------------------------
def evaluate_individual(args):
    pop_idx, seed_idx, weights, seed, rpc_host, rpc_port, authkey, max_steps = args
    model = NNPolicy(); set_weights_vector(model, weights); total_reward = 0.0
    try:
        env = gym.make("LunarLander-v3", render_mode="rgb_array")
        obs, _ = env.reset(seed=int(seed)); last_valid_state = np.zeros(7, dtype=np.float32)
        with RPCClient(rpc_host, rpc_port, authkey) as rpc:
            rpc.reset()
            for t in range(max_steps):
                frame = env.render();
                if frame is None: break
                frame_bgr = frame[..., ::-1]
                state = rpc.update_full(frame_bgr, t)
                if state is None: state = rpc.predict_only()
                if state is None: state = last_valid_state
                else: last_valid_state = state
                s = torch.tensor(state, dtype=torch.float32)
                with torch.no_grad(): act = int(torch.argmax(model(s)).item())
                obs, reward, done, truncated, info = env.step(act)
                total_reward += reward
                if done or truncated: break
        env.close()
        return pop_idx, seed_idx, total_reward
    except Exception:
        return pop_idx, seed_idx, -1000.0

# ----------------------------
# 6. Competitive Fitness Function (无变动)
# ----------------------------
def calculate_competitive_fitness(results_matrix: np.ndarray, success_threshold: float = 0.0) -> np.ndarray:
    population_size, num_seeds = results_matrix.shape
    success_matrix = results_matrix > success_threshold
    solvers_per_seed = success_matrix.sum(axis=0)
    seed_weights = 1.0 / (solvers_per_seed + 1e-9)
    competitive_scores = np.zeros(population_size)
    for i in range(population_size):
        solved_seeds_indices = np.where(success_matrix[i, :])[0]
        if len(solved_seeds_indices) > 0:
            competitive_scores[i] = seed_weights[solved_seeds_indices].sum()
    return competitive_scores

# ----------------------------
# 7. Logging & Plotting Utilities
# ----------------------------
# [MODIFIED] 增加一个 suffix 参数以保存两份模型
def save_best_model(run_dir: str, vec: np.ndarray, suffix: str = ""):
    os.makedirs(run_dir, exist_ok=True)
    np.savez(os.path.join(run_dir, f"best_model_vec{suffix}.npz"), weights=vec)
    m = NNPolicy(); set_weights_vector(m, vec)
    torch.save(m.state_dict(), os.path.join(run_dir, f"best_model{suffix}.pt"))

def save_metrics_csv(run_dir: str, gen: int, best: float, avg_top: float, worst: float):
    path = os.path.join(run_dir, "metrics.csv")
    header_needed = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if header_needed: f.write("generation,best_avg_reward,avg_top10_reward,worst_avg_reward\n")
        f.write(f"{gen},{best},{avg_top},{worst}\n")

def plot_reward_curve(run_dir: str, all_gens, all_best, all_avg, all_worst):
    if len(all_gens) == 0: return
    plt.figure(figsize=(10, 6)); plt.plot(all_gens, all_best, label="Best Individual Avg Reward", color="green", linewidth=2); plt.plot(all_gens, all_avg, label="Top 10% Avg Reward", color="blue", linestyle="--", alpha=0.8); plt.plot(all_gens, all_worst, label="Worst Individual Avg Reward", color="red", linestyle=":", alpha=0.6); plt.xlabel("Generation"); plt.ylabel("Average Reward over Test Seeds"); plt.title("GA Training Curve"); plt.grid(True, linestyle="--", alpha=0.4); plt.legend(); out = os.path.join(run_dir, "reward_curve.png"); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

# ----------------------------
# 8. Main GA Loop
# ----------------------------
def run_ga(population_size, generations, elite_frac, sigma, processes,
           rpc_host, rpc_port, authkey,
           outdir: str, tag: Optional[str],
           pool_size: int, subset_k: int, base_seed: int, 
           shuffle_pool: bool, pool_rng_seed: int, global_seed: Optional[int],
           resume: bool, max_steps: int, success_threshold: float,
           seed_refresh_frac: float, seed_refresh_direction: str,
           # [NEW] 接收新参数
           max_seed_age: int):

    if global_seed is not None: set_global_seed(global_seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ga_{ts}" if not tag else f"ga_{ts}_{tag}"
    run_dir = os.path.join(outdir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    cfg = {k:v for k,v in locals().items() if k not in ['pop', 'model', 'base_vec']}
    with open(os.path.join(run_dir, "config.json"), "w") as f: json.dump(cfg, f, indent=2)

    portfolio = SeedPortfolioManager(pool_size, base_seed, shuffle_pool, pool_rng_seed, subset_k)
    seed_age_path = os.path.join(run_dir, "seed_ages.json")
    if resume and os.path.exists(seed_age_path):
        try:
            with open(seed_age_path, "r") as f: portfolio.load_state_dict(json.load(f))
            print(f"🔁 [RESUME] Loaded seed ages from {seed_age_path}")
        except Exception as e:
            print(f"[WARN] Failed to load seed ages: {e}.")

    model = NNPolicy(); base_vec = get_weights_vector(model)
    pop = [mutate(base_vec, sigma=0.2) for _ in range(population_size)]

    best_global_reward = -np.inf; best_vec_global = None
    curve_gens, curve_best, curve_avg, curve_worst = [], [], [], []

    # --- 主训练循环 ---
    for gen in range(1, generations + 1):
        subset_seeds = portfolio.get_active_subset()
        jobs = [(i, j, pop[i], seed, rpc_host, rpc_port, authkey, max_steps) for i in range(population_size) for j, seed in enumerate(subset_seeds)]
        results_matrix = np.full((population_size, subset_k), -np.inf)

        try:
            with mp.Pool(processes=processes) as pool:
                results_iterator = tqdm(pool.imap_unordered(evaluate_individual, jobs), total=len(jobs), desc=f"Generation {gen}/{generations}")
                for pop_idx, seed_idx, reward in results_iterator:
                    results_matrix[pop_idx, seed_idx] = reward
        except Exception as e:
            print(f"\n[FATAL] Multiprocessing pool failed: {e}"); break

        # [MODIFIED] 传入 max_seed_age
        portfolio.update_and_refresh(results_matrix, seed_refresh_frac, seed_refresh_direction, success_threshold, max_seed_age)
        with open(seed_age_path, "w") as f: json.dump(portfolio.state_dict(), f)
        
        competitive_scores = calculate_competitive_fitness(results_matrix, success_threshold)
        elite_num = max(2, int(elite_frac * population_size)); elite_idx = np.argsort(competitive_scores)[-elite_num:]
        elites = [pop[i] for i in elite_idx]

        raw_avg_rewards = results_matrix.mean(axis=1)
        best_individual_idx = elite_idx[-1]
        best_reward = float(raw_avg_rewards[best_individual_idx])
        avg_top_reward = float(raw_avg_rewards[elite_idx].mean())
        worst_reward = float(np.min(raw_avg_rewards))
        
        print(f"🏆 [GEN {gen:03d}] BestAvgReward={best_reward:+.2f}  AvgTop{elite_num}={avg_top_reward:+.2f}  WorstAvgReward={worst_reward:+.2f}  Sigma={sigma:.3f}")

        curve_gens.append(gen); curve_best.append(best_reward); curve_avg.append(avg_top_reward); curve_worst.append(worst_reward)
        save_metrics_csv(run_dir, gen, best_reward, avg_top_reward, worst_reward)
        plot_reward_curve(run_dir, curve_gens, curve_best, curve_avg, curve_worst)

        if best_reward > best_global_reward:
            best_global_reward = best_reward
            best_vec_global = pop[best_individual_idx]
            # [MODIFIED] 保存为训练期间最佳
            save_best_model(run_dir, best_vec_global, suffix="_training_best")
            print(f"💾 [SAVE] New best model (training) to {run_dir} (avg reward {best_global_reward:.2f})")

        new_pop = elites.copy()
        while len(new_pop) < population_size:
            p1, p2 = random.sample(elites, 2); c1, c2 = uniform_crossover(p1, p2)
            new_pop.append(mutate(c1, sigma))
            if len(new_pop) < population_size: new_pop.append(mutate(c2, sigma))
        pop = new_pop
        sigma = max(0.01, sigma * 0.99)

    print("\n✅ Training finished.")

    # ----------------------------
    # [NEW] 9. Final Full Evaluation
    # ----------------------------
    print(f"\n🏁 Starting final evaluation on full {pool_size}-seed pool...")
    master_pool = portfolio.master_pool
    final_jobs = [(i, j, pop[i], seed, rpc_host, rpc_port, authkey, max_steps) for i in range(population_size) for j, seed in enumerate(master_pool)]
    
    final_results_matrix = np.full((population_size, len(master_pool)), -np.inf)
    
    try:
        with mp.Pool(processes=processes) as pool:
            results_iterator = tqdm(pool.imap_unordered(evaluate_individual, final_jobs), total=len(final_jobs), desc="Final Full Evaluation")
            for pop_idx, seed_idx, reward in results_iterator:
                final_results_matrix[pop_idx, seed_idx] = reward
    except Exception as e:
        print(f"\n[FATAL] Final evaluation failed: {e}")

    # 计算最终总决赛的平均分
    final_avg_scores = final_results_matrix.mean(axis=1)
    best_final_idx = np.argmax(final_avg_scores)
    best_final_score = final_avg_scores[best_final_idx]
    best_final_vec = pop[best_final_idx]

    print(f"🏆 [Final Result] Best model from full {pool_size}-seed eval: {best_final_score:+.2f} (Individual {best_final_idx})")
    
    # [NEW] 保存总决赛最佳模型
    save_best_model(run_dir, best_final_vec, suffix="_full_eval")
    print(f"💾 [SAVE] Best model (full eval) saved to {run_dir}")
    
    # 也可以保存完整的总决赛分数矩阵
    np.savez(os.path.join(run_dir, "final_evaluation_matrix.npz"), results=final_results_matrix)

# ----------------------------
# 10. CLI (Command Line Interface)
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="GA Client with Competitive Fitness and Dual-Refresh Strategy")
    # RPC
    parser.add_argument("--rpc-host", type=str, default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=6001)
    parser.add_argument("--authkey", type=str, default="yolo-rpc")
    # GA Core
    parser.add_argument("--processes", type=int, default=8, help="Number of parallel workers.")
    parser.add_argument("--population", type=int, default=120)
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument("--elite-frac", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.1)
    
    # [MODIFIED] 更新了默认值
    # Seed Portfolio
    parser.add_argument("--pool-size", type=int, default=256, help="Total size of the master seed pool.")
    parser.add_argument("--subset-k", type=int, default=5, help="Number of seeds to test each generation.")
    parser.add_argument("--base-seed", type=int, default=202510, help="Seed to generate the master seed pool.")
    parser.add_argument("--shuffle-pool", action="store_true", default=True)
    parser.add_argument("--no-shuffle-pool", dest="shuffle_pool", action="store_false")
    parser.add_argument("--pool-rng-seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1000)
    
    # [MODIFIED] 更新了默认值
    # Seed Refresh
    parser.add_argument("--seed-refresh-frac", type=float, default=0.4, help="Fraction of seed subset to refresh each gen (0.0 to disable).")
    parser.add_argument("--seed-refresh-direction", type=str, default="bottom", choices=['none', 'top', 'bottom'], help="Refresh top (easiest) or bottom (hardest) seeds.")
    # [NEW] 新增年龄替换参数
    parser.add_argument("--max-seed-age", type=int, default=10, help="Max age for a seed before forced replacement (0 to disable).")
    
    # Competitive Fitness
    parser.add_argument("--success-threshold", type=float, default=0.0, help="Reward threshold to consider a seed 'solved'.")
    # Logging & Reproducibility
    parser.add_argument("--global-seed", type=int, default=42, help="Global seed for reproducibility.")
    parser.add_argument("--outdir", type=str, default="runs_ga_final", help="Directory for logs and models.")
    parser.add_argument("--tag", type=str, default="", help="Optional tag for run folder name.")
    parser.add_argument("--resume", action="store_true", help="Resume seed ages from run folder.")
    args = parser.parse_args()

    print(f"🚀 [INFO] Starting GA with Competitive Fitness & Dual-Refresh Strategy")
    print(f"📡 Connecting to RPC server at {args.rpc_host}:{args.rpc_port}")

    run_ga(
        population_size=args.population, 
        generations=args.generations, 
        elite_frac=args.elite_frac,
        sigma=args.sigma, 
        processes=args.processes, 
        rpc_host=args.rpc_host,
        rpc_port=args.rpc_port, 
        authkey=args.authkey, 
        outdir=args.outdir, 
        tag=(args.tag or None),
        pool_size=args.pool_size, 
        subset_k=args.subset_k, 
        base_seed=args.base_seed,
        shuffle_pool=args.shuffle_pool, 
        pool_rng_seed=args.pool_rng_seed,
        global_seed=args.global_seed, 
        resume=args.resume, 
        max_steps=args.max_steps,
        success_threshold=args.success_threshold,
        seed_refresh_frac=args.seed_refresh_frac,
        seed_refresh_direction=args.seed_refresh_direction,
        # [NEW] 传入新参数
        max_seed_age=args.max_seed_age
    )

if __name__ == "__main__":
    main()