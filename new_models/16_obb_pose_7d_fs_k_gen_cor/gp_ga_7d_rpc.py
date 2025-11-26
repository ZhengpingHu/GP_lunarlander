#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# LunarLander GA Client: Proportional Fitness Sharing (No Thresholds) + Corrected Metrics

import os
import argparse
import random
import multiprocessing as mp
from multiprocessing.connection import Client
from typing import Optional, Tuple, Sequence, List, Dict
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from tqdm import tqdm
import pickle

# [PLOT] 离屏绘图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Boilerplate ---
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
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ----------------------------
# 2. Neural Network
# ----------------------------
class NNPolicy(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        # LunarLander 输入7维，输出4维
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
# 3. Seed Management
# ----------------------------
class SeedAgeScheduler:
    def __init__(self, seed_pool: Sequence[int], rng_seed: int = 0):
        self.seed_pool = list(seed_pool)
        self.ages: List[int] = [0 for _ in self.seed_pool]
        self.rng = random.Random(rng_seed)
    @classmethod
    def from_fixed_pool(cls, pool_size: int, base_seed: int, shuffle: bool, rng_seed: int):
        g = np.random.default_rng(base_seed)
        pool = g.integers(low=0, high=2**31-1, size=pool_size, dtype=np.int32).tolist()
        if shuffle:
            rr = random.Random(rng_seed); rr.shuffle(pool)
        return cls(pool, rng_seed=rng_seed)
    def select_subset(self, k: int) -> List[int]:
        n = len(self.seed_pool); idx = list(range(n)); self.rng.shuffle(idx)
        idx.sort(key=lambda i: self.ages[i], reverse=True); return [self.seed_pool[i] for i in idx[:k]]
    def update_after_generation(self, chosen_seeds: Sequence[int]):
        chosen = set(chosen_seeds)
        for i, s in enumerate(self.seed_pool): self.ages[i] = 0 if s in chosen else self.ages[i] + 1
    def get_age_of_seed(self, seed: int) -> int:
        try: return self.ages[self.seed_pool.index(seed)]
        except ValueError: return -1
    def state_dict(self) -> dict: return {"seed_pool": self.seed_pool, "ages": self.ages}
    def load_state_dict(self, d: dict): self.ages = list(d["ages"])

class SeedPortfolioManager:
    def __init__(self, pool_size, base_seed, shuffle, pool_rng_seed, subset_k):
        self.scheduler = SeedAgeScheduler.from_fixed_pool(pool_size, base_seed, shuffle, pool_rng_seed)
        self.master_pool = self.scheduler.seed_pool
        self.active_subset = self.scheduler.select_subset(subset_k)
        print(f"🌱 Initial seed subset: {self.active_subset}")
    def get_active_subset(self) -> List[int]: return self.active_subset
    
    def update_and_refresh(self, results_matrix: np.ndarray, refresh_frac: float, refresh_direction: str, max_seed_age: int):
        # Note: Removed success_threshold argument because we use raw scores now
        self.scheduler.update_after_generation(self.active_subset)
        indices_to_replace = set()
        
        # 1. Age Refresh
        if max_seed_age > 0:
            age_indices = {i for i, s in enumerate(self.active_subset) if self.scheduler.get_age_of_seed(s) > max_seed_age}
            indices_to_replace.update(age_indices)
            
        # 2. Performance Refresh (Based on Total Score Difficulty)
        # Lower sum = Harder seed; Higher sum = Easier seed
        seed_total_scores = results_matrix.sum(axis=0)
        num_perf_replace = int(np.floor(len(self.active_subset) * refresh_frac))
        
        if num_perf_replace > 0 and refresh_direction != 'none':
            sorted_indices = np.argsort(seed_total_scores)
            perf_indices = set()
            if refresh_direction == 'bottom': # Replace Easiest
                indices = sorted_indices[-num_perf_replace:]
                perf_indices.update(indices)
            elif refresh_direction == 'top': # Replace Hardest
                indices = sorted_indices[:num_perf_replace]
                perf_indices.update(indices)
            indices_to_replace.update(perf_indices)
            
        if not indices_to_replace: return
        
        num_to_replace = len(indices_to_replace)
        candidate_pool = [s for s in self.master_pool if s not in self.active_subset]
        if len(candidate_pool) < num_to_replace:
            num_to_replace = len(candidate_pool); indices_to_replace = list(indices_to_replace)[:num_to_replace]
        
        if num_to_replace > 0:
            new_seeds = random.sample(candidate_pool, num_to_replace)
            for i_rep, i_new in zip(indices_to_replace, range(num_to_replace)):
                self.active_subset[i_rep] = new_seeds[i_new]
            print(f"  - Replaced {len(new_seeds)} seeds.")

    def state_dict(self): return self.scheduler.state_dict()
    def load_state_dict(self, d): self.scheduler.load_state_dict(d)

# ----------------------------
# 4. RPC Client
# ----------------------------
class RPCClient:
    def __init__(self, host, port, authkey="yolo-rpc"):
        self.address = (host, port); self.authkey = authkey.encode("utf-8"); self.conn: Optional[Client] = None
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
# 5. Evaluation Function
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
# 6. Fitness Logic (PROPORTIONAL SHARING - NO THRESHOLDS)
# ----------------------------
def calculate_competitive_fitness(results_matrix: np.ndarray) -> np.ndarray:
    """
    [CORRECTED] Proportional Fitness Sharing.
    Formula: G'(i, k) = G(i, k) / SUM_j(G(j, k))
    
    CRITICAL FIX FOR LUNARLANDER:
    LunarLander rewards can be negative. Proportional sharing requires positive values.
    We shift the entire matrix so the minimum value is a small positive number.
    """
    # 1. Shift to Positive
    min_val = np.min(results_matrix)
    if min_val <= 0:
        # Shift everything so the worst person gets 0.1 (to avoid zero division)
        shifted_matrix = results_matrix - min_val + 0.1
    else:
        shifted_matrix = results_matrix

    # 2. Calculate Total Rewards per Seed (Denominator)
    # Sum down the columns (across population)
    total_rewards_per_seed = shifted_matrix.sum(axis=0)
    
    # Safety: Avoid division by zero
    total_rewards_per_seed[total_rewards_per_seed == 0] = 1e-9
    
    # 3. Calculate Shared Fitness Matrix
    # Each individual's score divided by the total score for that seed
    shared_fitness_matrix = shifted_matrix / total_rewards_per_seed
    
    # 4. Sum across seeds to get final fitness
    fitness_scores = shared_fitness_matrix.sum(axis=1)
            
    return fitness_scores

# ----------------------------
# 7. Metrics Logging & Plotting (Separated Plots)
# ----------------------------
def save_metrics_csv(run_dir, gen, metrics):
    path = os.path.join(run_dir, "metrics.csv")
    header_needed = not os.path.exists(path)
    df = pd.DataFrame([metrics])
    df.to_csv(path, mode='a', header=header_needed, index=False)

def plot_separated_curves(run_dir, df_history):
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    gens = df_history['generation']
    
    # Plot 1: Fitness Score
    plt.figure(figsize=(10, 6))
    plt.plot(gens, df_history['best_fitness_score'], label='Best Fitness (Shared)', color='purple')
    plt.xlabel("Generation"); plt.ylabel("Shared Fitness"); plt.title("Optimization Metric")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "plot1_fitness_score.png")); plt.close()

    # Plot 2: Global Max Raw Reward
    plt.figure(figsize=(10, 6))
    plt.plot(gens, df_history['global_max_raw_reward'], label='Pop Max Raw', color='green')
    plt.plot(gens, df_history['global_avg_raw_reward'], label='Pop Avg Raw', color='gray', linestyle='--')
    plt.xlabel("Generation"); plt.ylabel("Raw Reward"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "plot2_max_raw_reward.png")); plt.close()

    # Plot 3: Fitness vs Selected
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:purple'
    ax1.set_xlabel('Generation'); ax1.set_ylabel('Best Fitness', color=color)
    ax1.plot(gens, df_history['best_fitness_score'], color=color, label='Fitness')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Selected Individual Raw', color=color)
    ax2.plot(gens, df_history['selected_individual_reward'], color=color, linestyle='--', label='Selected Raw')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title("Metric Comparison"); fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot3_fitness_vs_reward.png")); plt.close()

def plot_final_violin(final_results_matrix, run_dir):
    try:
        avg_scores_per_ind = final_results_matrix.mean(axis=1)
        plt.figure(figsize=(10, 6))
        sns.violinplot(y=avg_scores_per_ind, inner='quartile', color='lightblue')
        plt.title('Final Population Performance Distribution')
        plt.ylabel('Average Reward per Individual')
        plt.grid(True, axis='y', alpha=0.5)
        plt.savefig(os.path.join(run_dir, "final_violin_plot.png")); plt.close()
    except Exception: pass

def save_history_snapshot(run_dir, gen, population, results_matrix, subset_seeds, fitness_scores):
    history_dir = os.path.join(run_dir, "history")
    os.makedirs(history_dir, exist_ok=True)
    filename = os.path.join(history_dir, f"gen_{gen:04d}.npz")
    np.savez_compressed(filename, generation=gen, population_weights=np.array(population), 
                        results_matrix=results_matrix, subset_seeds=np.array(subset_seeds), 
                        fitness_scores=fitness_scores)

def save_full_checkpoint(run_dir, gen, population, portfolio, fitness_history, args):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {"generation": gen, "population": population, "portfolio_state": portfolio.state_dict(),
             "fitness_history": fitness_history, "args": vars(args)}
    torch.save(state, os.path.join(ckpt_dir, f"checkpoint_gen_{gen:04d}.pt"))

# ----------------------------
# 8. Main GA Loop
# ----------------------------
def run_ga(args):
    if args.global_seed: set_global_seed(args.global_seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.outdir, f"ga_lunar_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    
    portfolio = SeedPortfolioManager(args.pool_size, args.base_seed, args.shuffle_pool, 
                                     args.pool_rng_seed, args.subset_k)
    
    model = NNPolicy(); base_vec = get_weights_vector(model)
    pop = [mutate(base_vec, args.sigma) for _ in range(args.population)]
    
    fitness_history = {}; history_records = []; best_global_raw = -np.inf
    print(f"🚀 Training started. Logs at: {run_dir}")

    for gen in range(1, args.generations + 1):
        subset_seeds = portfolio.get_active_subset()
        jobs = []
        for i in range(args.population):
            for j, seed in enumerate(subset_seeds):
                jobs.append((i, j, pop[i], seed, args.rpc_host, args.rpc_port, args.authkey, args.max_steps))
        
        results_matrix = np.full((args.population, args.subset_k), -np.inf)
        with mp.Pool(processes=args.processes) as pool:
            for pop_idx, seed_idx, reward in tqdm(pool.imap_unordered(evaluate_individual, jobs), 
                                                 total=len(jobs), desc=f"Gen {gen}"):
                results_matrix[pop_idx, seed_idx] = reward

        # 1. Update & Refresh (Uses Raw Score Sum, No Threshold needed)
        portfolio.update_and_refresh(results_matrix, args.seed_refresh_frac, args.seed_refresh_direction, args.max_seed_age)
        
        # 2. Calc Raw Stats
        raw_avg_rewards_per_ind = results_matrix.mean(axis=1)
        global_max_raw = np.max(raw_avg_rewards_per_ind) # Population Absolute Max
        global_avg_raw = np.mean(raw_avg_rewards_per_ind)
        
        # 3. Calc Fitness (PROPORTIONAL SHARING, NO THRESHOLD)
        comp_scores = calculate_competitive_fitness(results_matrix)
        
        # 4. Smooth Fitness
        smoothed_scores = np.zeros(args.population)
        new_hist = {}
        for i in range(args.population):
            k = tuple(pop[i])
            hist = (fitness_history.get(k, []) + [comp_scores[i]])[-args.fitness_avg_generations:]
            smoothed_scores[i] = np.mean(hist)
            new_hist[k] = hist
        fitness_history = new_hist
        
        # 5. Selection
        elite_num = max(2, int(args.elite_frac * args.population))
        sorted_indices = np.argsort(smoothed_scores)
        elite_indices = sorted_indices[-elite_num:]
        
        champion_idx = elite_indices[-1]
        best_fitness_val = smoothed_scores[champion_idx]
        selected_individual_raw_reward = raw_avg_rewards_per_ind[champion_idx]
        
        print(f"🏆 [GEN {gen}] PopMax={global_max_raw:.1f} | BestFit={best_fitness_val:.4f} | SelRaw={selected_individual_raw_reward:.1f}")
        
        # 6. Logging
        metrics = {
            "generation": gen,
            "best_fitness_score": best_fitness_val,
            "global_max_raw_reward": global_max_raw,
            "global_avg_raw_reward": global_avg_raw,
            "selected_individual_reward": selected_individual_raw_reward
        }
        history_records.append(metrics)
        save_metrics_csv(run_dir, gen, metrics)
        save_history_snapshot(run_dir, gen, pop, results_matrix, subset_seeds, smoothed_scores)
        
        if gen % args.checkpoint_freq == 0:
            save_full_checkpoint(run_dir, gen, pop, portfolio, fitness_history, args)
        if gen % 1 == 0:
            df_hist = pd.DataFrame(history_records)
            plot_separated_curves(run_dir, df_hist)

        if selected_individual_raw_reward > best_global_raw:
            best_global_raw = selected_individual_raw_reward
            np.savez(os.path.join(run_dir, "best_model.npz"), weights=pop[champion_idx])

        # 7. Evolution
        elites = [pop[i] for i in elite_indices]
        new_pop = elites.copy()
        while len(new_pop) < args.population:
            p1, p2 = random.sample(elites, 2); c1, c2 = uniform_crossover(p1, p2)
            new_pop.append(mutate(c1, args.sigma))
            if len(new_pop) < args.population: new_pop.append(mutate(c2, args.sigma))
        pop = new_pop

    print("\n✅ Training finished.")

    # Final Full Evaluation
    print(f"\n🏁 Starting final evaluation on full {len(portfolio.master_pool)}-seed pool...")
    master_pool = portfolio.master_pool
    final_jobs = []
    for i in range(args.population):
        for j, seed in enumerate(master_pool):
            final_jobs.append((i, j, pop[i], seed, args.rpc_host, args.rpc_port, args.authkey, args.max_steps))
            
    final_results_matrix = np.full((args.population, len(master_pool)), -np.inf)
    try:
        with mp.Pool(processes=args.processes) as pool:
            results_iterator = tqdm(pool.imap_unordered(evaluate_individual, final_jobs), 
                                  total=len(final_jobs), desc="Final Eval")
            for pop_idx, seed_idx, reward in results_iterator:
                final_results_matrix[pop_idx, seed_idx] = reward
    except Exception as e:
        print(f"\n[FATAL] Final evaluation failed: {e}")

    final_avg_scores = final_results_matrix.mean(axis=1)
    best_final_idx = np.argmax(final_avg_scores)
    print(f"🏆 [Final Result] Best Robust Model: {final_avg_scores[best_final_idx]:+.2f}")
    np.savez(os.path.join(run_dir, "best_model_full_eval.npz"), weights=pop[best_final_idx])
    
    plot_final_violin(final_results_matrix, run_dir)
    print(f"💾 Results saved to {run_dir}")

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpc-host", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=6001) # Port 6001 for LunarLander Server
    parser.add_argument("--authkey", default="yolo-rpc")
    
    parser.add_argument("--population", type=int, default=120)
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    
    # Seed Portfolio
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument("--subset-k", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=202510)
    
    # Dual Refresh
    parser.add_argument("--seed-refresh-frac", type=float, default=0.4)
    parser.add_argument("--seed-refresh-direction", type=str, default="bottom")
    parser.add_argument("--max-seed-age", type=int, default=10)
    
    parser.add_argument("--elite-frac", type=float, default=0.25)
    parser.add_argument("--sigma", type=float, default=0.1)
    # Note: success_threshold is REMOVED from Fitness calculation logic entirely
    # But we keep the arg just in case you have legacy scripts calling it (it won't be used)
    parser.add_argument("--success-threshold", type=float, default=0.0) 
    parser.add_argument("--fitness-avg-generations", type=int, default=5)
    
    parser.add_argument("--shuffle-pool", action="store_true", default=True)
    parser.add_argument("--pool-rng-seed", type=int, default=42)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--outdir", default="runs_ga_final")
    parser.add_argument("--checkpoint-freq", type=int, default=5)

    args = parser.parse_args()
    run_ga(args)