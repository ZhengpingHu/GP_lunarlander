#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Final NEAT Version: Competitive Fitness + Dynamic Seed Refresh + Speciation

import os
import argparse
import random
import multiprocessing as mp
from multiprocessing.connection import Client
from typing import Optional, Tuple, Sequence, List, Dict
import json
from datetime import datetime
import numpy as np
import torch # 仅用于 torch.tensor
import gymnasium as gym
import cv2
from tqdm import tqdm
import pickle # [NEW] 用于保存 NEAT 基因组

# [NEW] 导入 NEAT
import neat

# [PLOT] 离屏绘图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
    # [MODIFIED] NEAT 主要使用 random 和 numpy
    # torch.manual_seed(seed)
    # if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ----------------------------
# (REMOVED: NNPolicy, get_weights_vector, set_weights_vector, uniform_crossover, mutate)
# ----------------------------

# ----------------------------
# 2. Seed Management System (无变动)
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
    def load_state_dict(self, d: dict):
        if d["seed_pool"] != self.seed_pool: raise ValueError("Seed pool mismatch.")
        self.ages = list(d["ages"])

class SeedPortfolioManager:
    def __init__(self, pool_size, base_seed, shuffle, pool_rng_seed, subset_k):
        self.scheduler = SeedAgeScheduler.from_fixed_pool(pool_size, base_seed, shuffle, pool_rng_seed)
        self.master_pool = self.scheduler.seed_pool
        self.active_subset = self.scheduler.select_subset(subset_k)
        print(f"🌱 Initial seed subset: {self.active_subset}")
    def get_active_subset(self) -> List[int]: return self.active_subset
    def update_and_refresh(self, results_matrix: np.ndarray, refresh_frac: float, refresh_direction: str, success_threshold: float, max_seed_age: int):
        self.scheduler.update_after_generation(self.active_subset)
        indices_to_replace = set()
        if max_seed_age > 0:
            age_indices_to_replace = {i for i, s in enumerate(self.active_subset) if self.scheduler.get_age_of_seed(s) > max_seed_age}
            if age_indices_to_replace:
                print(f"Replacing {len(age_indices_to_replace)} seeds due to max age ({max_seed_age}).")
                indices_to_replace.update(age_indices_to_replace)
        num_perf_replace = int(np.floor(len(self.active_subset) * refresh_frac))
        if num_perf_replace > 0 and refresh_direction != 'none':
            success_matrix = results_matrix > success_threshold; solvers_per_seed = success_matrix.sum(axis=0)
            seed_difficulty_scores = 1.0 / (solvers_per_seed + 1e-9); sorted_indices_by_difficulty = np.argsort(seed_difficulty_scores)
            perf_indices_to_replace = set()
            if refresh_direction == 'bottom':
                indices = sorted_indices_by_difficulty[-num_perf_replace:]; perf_indices_to_replace.update(indices)
                print(f"Replacing {len(perf_indices_to_replace)} hardest performing seeds (highest weights).")
            elif refresh_direction == 'top':
                indices = sorted_indices_by_difficulty[:num_perf_replace]; perf_indices_to_replace.update(indices)
                print(f"Replacing {len(perf_indices_to_replace)} easiest performing seeds (lowest weights).")
            indices_to_replace.update(perf_indices_to_replace)
        if not indices_to_replace: return
        num_to_replace_total = len(indices_to_replace)
        old_seeds_to_replace = {self.active_subset[i] for i in indices_to_replace}
        candidate_pool = [s for s in self.master_pool if s not in self.active_subset]
        if len(candidate_pool) < num_to_replace_total:
            print(f"[WARN] Not enough ({len(candidate_pool)}) unused seeds to replace {num_to_replace_total}.")
            num_to_replace_total = len(candidate_pool); indices_to_replace = list(indices_to_replace)[:num_to_replace_total]
        if num_to_replace_total == 0: print("[WARN] No seeds to replace."); return
        new_seeds = random.sample(candidate_pool, num_to_replace_total)
        for i_replace, i_new in zip(indices_to_replace, range(num_to_replace_total)):
            self.active_subset[i_replace] = new_seeds[i_new]
        print(f"  - Replaced: {list(old_seeds_to_replace)}"); print(f"  + New active subset: {self.active_subset}")
    def state_dict(self): return self.scheduler.state_dict()
    def load_state_dict(self, d): self.scheduler.load_state_dict(d)

# ----------------------------
# 3. RPC Client (无变动)
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
# 4. [MODIFIED] NEAT Evaluation Function (Worker)
# ----------------------------
def evaluate_individual_neat(args):
    # [MODIFIED] 接收 genome 和 config, 而不是 weights
    genome_id, genome, config, seed_idx, seed, rpc_host, rpc_port, authkey, max_steps = args
    
    # [MODIFIED] 从 genome 和 config 创建神经网络
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    total_reward = 0.0
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
                
                # [MODIFIED] 使用 NEAT 的 net.activate()，它接收 list 或 numpy 数组
                output = net.activate(state)
                act = int(np.argmax(output)) # NEAT 的输出是浮点数列表
                
                obs, reward, done, truncated, info = env.step(act)
                total_reward += reward
                if done or truncated: break
        env.close()
        # [MODIFIED] 返回 genome_id
        return genome_id, seed_idx, total_reward
    except Exception:
        return genome_id, seed_idx, -1000.0

# ----------------------------
# 5. Competitive Fitness Function (无变动)
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
# 6. Logging & Plotting Utilities
# ----------------------------
# [MODIFIED] 保存 .pkl 基因组
def save_best_genome(run_dir: str, genome: neat.DefaultGenome, suffix: str = ""):
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, f"best_genome{suffix}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(genome, f)
    # 我们可以额外保存一个网络图
    try:
        from neat.visualize import draw_net
        node_names = {-1:'x', -2:'y', -3:'theta', -4:'vx', -5:'vy', -6:'v_theta', -7:'speed',
                       0:'No-Op', 1:'Left', 2:'Main', 3:'Right'}
        draw_net(config, genome, view=False, filename=os.path.join(run_dir, f"best_genome_net{suffix}.gv"),
                 node_names=node_names, show_disabled=True)
        print(f"  ...Network visualization saved to .gv file.")
    except Exception as e:
        print(f"  ...Could not visualize network (is graphviz installed?): {e}")

def save_metrics_csv(run_dir: str, gen: int, best: float, avg_top: float, worst: float):
    path = os.path.join(run_dir, "metrics.csv")
    header_needed = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if header_needed: f.write("generation,best_avg_reward,avg_top_reward,worst_avg_reward\n")
        f.write(f"{gen},{best},{avg_top},{worst}\n")

def plot_reward_curve(run_dir: str, all_gens, all_best, all_avg, all_worst, elite_frac: float):
    if len(all_gens) == 0: return
    plt.figure(figsize=(10, 6)); plt.plot(all_gens, all_best, label="Best Individual Avg Reward", color="green", linewidth=2); plt.plot(all_gens, all_avg, label=f"Top {int(elite_frac*100)}% Avg Reward", color="blue", linestyle="--", alpha=0.8); plt.plot(all_gens, all_worst, label="Worst Individual Avg Reward", color="red", linestyle=":", alpha=0.6); plt.xlabel("Generation"); plt.ylabel("Average Reward over Test Seeds"); plt.title("GA Training Curve"); plt.grid(True, linestyle="--", alpha=0.4); plt.legend(); out = os.path.join(run_dir, "reward_curve.png"); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

def plot_final_summary_plots(final_results_matrix: np.ndarray, master_pool: List[int], run_dir: str):
    try:
        print("📊 Generating final evaluation plots...")
        avg_scores_per_individual = final_results_matrix.mean(axis=1)
        avg_scores_per_seed = final_results_matrix.mean(axis=0)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        sns.violinplot(y=avg_scores_per_individual, ax=ax1, inner='quartile', color='lightblue'); ax1.set_title(f'Distribution of Individual Performance\n(N={len(avg_scores_per_individual)} Individuals)'); ax1.set_ylabel(f'Average Reward (over {len(master_pool)} seeds)'); ax1.grid(True, linestyle="--", alpha=0.5)
        sns.violinplot(y=avg_scores_per_seed, ax=ax2, inner='quartile', color='lightgreen'); ax2.set_title(f'Distribution of Seed Difficulty\n(N={len(master_pool)} Seeds)'); ax2.set_ylabel(f'Average Reward (across {len(avg_scores_per_individual)} individuals)'); ax2.grid(True, linestyle="--", alpha=0.5)
        bins = [-np.inf, 0, 200, np.inf]; labels = ["Fail (< 0)", "Pass (0-200)", "Success (>= 200)"]
        categories = pd.cut(avg_scores_per_seed, bins=bins, labels=labels, right=False)
        proportions = categories.value_counts(normalize=True).sort_index() * 100
        prop_text = "Seed Pool Proportions:\n"; 
        for name, pct in proportions.items(): prop_text += f"  - {name}: {pct:.1f}%\n"
        fig.text(0.5, 0.01, prop_text, ha='center', fontsize=10, wrap=True)
        plt.tight_layout(rect=[0, 0.05, 1, 1]); plot_path = os.path.join(run_dir, "final_evaluation_plots.png")
        plt.savefig(plot_path, dpi=150); plt.close(fig)
        print(f"📈 Final summary plots saved to {plot_path}")
    except Exception as e:
        print(f"[WARN] Failed to generate final plots: {e}")

# ----------------------------
# 7. [MODIFIED] Main NEAT Loop
# ----------------------------
def run_neat(processes,
           rpc_host, rpc_port, authkey,
           outdir: str, tag: Optional[str],
           pool_size: int, subset_k: int, base_seed: int, 
           shuffle_pool: bool, pool_rng_seed: int, global_seed: Optional[int],
           resume: bool, max_steps: int, success_threshold: float,
           seed_refresh_frac: float, seed_refresh_direction: str,
           max_seed_age: int,
           # [NEW] NEAT 特有参数
           config_file: str, generations: int):

    if global_seed is not None: set_global_seed(global_seed)

    # --- 设置日志目录 ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"neat_{ts}" if not tag else f"neat_{ts}_{tag}"
    run_dir = os.path.join(outdir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # --- [NEW] 加载 NEAT 配置 ---
    global config # 设为全局，以便 worker 进程中的 `evaluate_individual_neat` 可以访问
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    # 保存所有配置参数
    cfg = {k:v for k,v in locals().items() if k not in ['config']}
    with open(os.path.join(run_dir, "config.json"), "w") as f: json.dump(cfg, f, indent=2)

    # --- 设置种子管理器 ---
    portfolio = SeedPortfolioManager(pool_size, base_seed, shuffle_pool, pool_rng_seed, subset_k)
    seed_age_path = os.path.join(run_dir, "seed_ages.json")
    if resume and os.path.exists(seed_age_path):
        try:
            with open(seed_age_path, "r") as f: portfolio.load_state_dict(json.load(f))
            print(f"🔁 [RESUME] Loaded seed ages from {seed_age_path}")
        except Exception as e:
            print(f"[WARN] Failed to load seed ages: {e}.")

    # --- 日志列表 ---
    best_global_reward = -np.inf
    curve_gens, curve_best, curve_avg, curve_worst = [], [], [], []
    
    # --- [NEW] 定义 NEAT 的评估函数 (eval_genomes) ---
    # 我们将其定义为内部函数，以利用闭包访问 portfolio, run_dir, processes 等变量
    def eval_genomes(genomes, config):
        # 1. 获取当前激活的种子
        subset_seeds = portfolio.get_active_subset()
        
        # 2. 创建作业列表
        jobs = []
        genome_id_to_pop_idx_map = {} # 映射 genome.key 到 results_matrix 的行索引
        pop_idx = 0
        for genome_id, genome in genomes:
            genome.fitness = -np.inf # 默认适应度
            genome_id_to_pop_idx_map[genome_id] = pop_idx
            for seed_idx, seed in enumerate(subset_seeds):
                jobs.append((genome_id, genome, config, seed_idx, seed, rpc_host, rpc_port, authkey, max_steps))
            pop_idx += 1
            
        results_matrix = np.full((len(genomes), subset_k), -np.inf)

        # 3. 运行并行评估
        try:
            with mp.Pool(processes=processes) as pool:
                results_iterator = tqdm(pool.imap_unordered(evaluate_individual_neat, jobs), total=len(jobs), desc=f"Generation {p.generation+1}/{generations}")
                for genome_id, seed_idx, reward in results_iterator:
                    pop_idx = genome_id_to_pop_idx_map[genome_id]
                    results_matrix[pop_idx, seed_idx] = reward
        except Exception as e:
            print(f"\n[FATAL] Multiprocessing pool failed: {e}")
            raise # 停止 NEAT 运行

        # 4. 计算竞争性适应度
        competitive_scores = calculate_competitive_fitness(results_matrix, success_threshold)

        # 5. 将适应度赋回给 NEAT
        for i, (genome_id, genome) in enumerate(genomes):
            genome.fitness = competitive_scores[i]

        # 6. 更新种子组合
        portfolio.update_and_refresh(results_matrix, seed_refresh_frac, seed_refresh_direction, success_threshold, max_seed_age)
        with open(seed_age_path, "w") as f: json.dump(portfolio.state_dict(), f)
        
        # 7. [NEW] 手动记录和绘图 (替代 N 代平滑，但保留日志)
        gen = p.generation + 1
        raw_avg_rewards = results_matrix.mean(axis=1)
        
        # 找到竞争力最强的个体
        best_genome_idx_by_fitness = np.argmax(competitive_scores)
        best_individual_idx = best_genome_idx_by_fitness # 我们的精英
        
        best_reward = float(raw_avg_rewards[best_individual_idx])
        
        # 找到原始分最高的 Top 25% (根据 config)
        elite_frac = config.reproduction_config.elitism
        elite_num = max(2, int(elite_frac * len(genomes)))
        # 注意：这里我们只能基于竞争力得分来找到精英
        elite_indices = np.argsort(competitive_scores)[-elite_num:]
        avg_top_reward = float(raw_avg_rewards[elite_indices].mean())
        
        worst_reward = float(np.min(raw_avg_rewards))
        
        print(f"\n🏆 [GEN {gen:03d}] BestAvgReward(Comp.Winner)={best_reward:+.2f}  AvgTop{elite_num}(Comp.Elite)={avg_top_reward:+.2f}  WorstAvgReward={worst_reward:+.2f}")

        curve_gens.append(gen); curve_best.append(best_reward); curve_avg.append(avg_top_reward); curve_worst.append(worst_reward)
        save_metrics_csv(run_dir, gen, best_reward, avg_top_reward, worst_reward)
        plot_reward_curve(run_dir, curve_gens, curve_best, curve_avg, curve_worst, elite_frac)

        # 保存训练期间的最佳模型（基于原始分）
        # [MODIFIED] 我们保存竞争力最强的那个个体的基因组
        nonlocal best_global_reward
        if best_reward > best_global_reward:
            best_global_reward = best_reward
            best_genome_this_gen = genomes[best_individual_idx][1]
            save_best_genome(run_dir, best_genome_this_gen, suffix="_training_best")
            print(f"💾 [SAVE] New best training genome to {run_dir} (avg reward {best_global_reward:.2f})")

    # --- [NEW] NEAT 种群设置 ---
    p = neat.Population(config)
    
    # [NEW] 添加 NEAT 报告器 (打印冠军结构和统计)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # [NEW] 添加检查点 (保存冠军结构)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    p.add_reporter(neat.Checkpointer(generation_interval=5, filename_prefix=os.path.join(checkpoint_dir, 'neat-cp-')))
    
    # --- 运行 NEAT 进化 ---
    print(f"🚀 [INFO] Starting NEAT evolution for {generations} generations...")
    winner = p.run(eval_genomes, generations)

    # --- 进化结束 ---
    print("\n✅ Training finished.")
    
    # [NEW] 保存最终的冠军基因组
    print(f"💾 [SAVE] Saving final best genome (winner)...")
    save_best_genome(run_dir, winner, suffix="_final_winner")

    # ----------------------------
    # 9. Final Full Evaluation
    # ----------------------------
    print(f"\n🏁 Starting final evaluation on full {pool_size}-seed pool...")
    master_pool = portfolio.master_pool
    
    # [MODIFIED] 使用最终种群 (p.population) 进行评估
    final_pop_genomes = list(p.population.values())
    final_pop_ids = list(p.population.keys())
    
    final_jobs = []
    for i, genome in enumerate(final_pop_genomes):
        genome_id = genome.key
        for j, seed in enumerate(master_pool):
            final_jobs.append((genome_id, genome, config, j, seed, rpc_host, rpc_port, authkey, max_steps))
            
    final_results_matrix = np.full((len(final_pop_genomes), len(master_pool)), -np.inf)
    
    # [MODIFIED] 映射 genome_id 到矩阵的行
    genome_id_to_pop_idx_map = {gid: i for i, gid in enumerate(final_pop_ids)}
    
    try:
        with mp.Pool(processes=processes) as pool:
            results_iterator = tqdm(pool.imap_unordered(evaluate_individual_neat, final_jobs), total=len(final_jobs), desc="Final Full Evaluation")
            for genome_id, seed_idx, reward in results_iterator:
                pop_idx = genome_id_to_pop_idx_map[genome_id]
                final_results_matrix[pop_idx, seed_idx] = reward
    except Exception as e:
        print(f"\n[FATAL] Final evaluation failed: {e}")

    final_avg_scores = final_results_matrix.mean(axis=1)
    final_sorted_indices = np.argsort(final_avg_scores)[::-1] # 从高到低

    best_final_idx_in_pop = final_sorted_indices[0]
    best_final_score = final_avg_scores[best_final_idx_in_pop]
    best_final_genome = final_pop_genomes[best_final_idx_in_pop]

    print(f"🏆 [Final Result] Best model from full {pool_size}-seed eval: {best_final_score:+.2f} (Genome ID {best_final_genome.key})")
    
    save_best_genome(run_dir, best_final_genome, suffix="_full_eval")
    print(f"💾 [SAVE] Best model (full eval) saved to {run_dir}")
    
    # [MODIFIED] 保存详细的最终报告 (包含基因组结构)
    print(f"📝 Saving detailed final report...")
    detailed_report = []
    elite_frac_from_cfg = config.reproduction_config.elitism
    num_to_report = max(1, int(elite_frac_from_cfg * len(final_pop_genomes)))
    
    for rank, pop_idx in enumerate(final_sorted_indices[:num_to_report]):
        genome = final_pop_genomes[pop_idx]
        ind_scores = final_results_matrix[pop_idx, :]
        per_seed_results = {str(master_pool[j]): float(ind_scores[j]) for j in range(len(master_pool))}
        
        # 记录基因组结构
        genome_struct = {
            'nodes': {nid: (n.bias, n.response, n.activation, n.aggregation) for nid, n in genome.nodes.items()},
            'connections': {cid: (c.in_node_id, c.out_node_id, c.weight, c.enabled) for cid, c in genome.connections.items()}
        }
        
        detailed_report.append({
            "final_rank": rank + 1,
            "genome_id": genome.key,
            "avg_reward_full": float(final_avg_scores[pop_idx]),
            "genome_structure": genome_struct, # 保存详细结构
            "per_seed_rewards": per_seed_results
        })
    
    with open(os.path.join(run_dir, "final_report_detailed.json"), "w") as f:
        json.dump(detailed_report, f) # 保存非缩进的，文件可能很大
    print(f"💾 [SAVE] Detailed final report for top {num_to_report} individuals saved to {run_dir}")
    
    # [NEW] 生成并保存小提琴图
    plot_final_summary_plots(final_results_matrix, master_pool, run_dir)

# ----------------------------
# 10. CLI (Command Line Interface)
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="NEAT Client with Competitive Fitness and Dynamic Seed Refresh")
    # RPC
    parser.add_argument("--rpc-host", type=str, default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=6001)
    parser.add_argument("--authkey", type=str, default="yolo-rpc")
    
    # [MODIFIED] NEAT Core Args
    parser.add_argument("--neat-config-file", type=str, default="neat.cfg", help="Path to the NEAT configuration file.")
    parser.add_argument("--generations", type=int, default=150, help="Number of generations to run.")
    parser.add_argument("--processes", type=int, default=8, help="Number of parallel workers.")

    # Seed Portfolio
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument("--subset-k", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=202510)
    parser.add_argument("--shuffle-pool", action="store_true", default=True)
    parser.add_argument("--no-shuffle-pool", dest="shuffle_pool", action="store_false")
    parser.add_argument("--pool-rng-seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1000)
    
    # Seed Refresh
    parser.add_argument("--seed-refresh-frac", type=float, default=0.4)
    parser.add_argument("--seed-refresh-direction", type=str, default="bottom", choices=['none', 'top', 'bottom'])
    parser.add_argument("--max-seed-age", type=int, default=10)
    
    # Competitive Fitness
    parser.add_argument("--success-threshold", type=float, default=0.0)

    # Logging & Reproducibility
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="runs_neat_final")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    
    args = parser.parse_args()

    print(f"🚀 [INFO] Starting NEAT with Competitive Fitness & Dual-Refresh Strategy")
    print(f"📡 Connecting to RPC server at {args.rpc_host}:{args.rpc_port}")

    run_neat(
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
        max_seed_age=args.max_seed_age,
        # [NEW] 传入 NEAT 参数
        config_file=args.neat_config_file,
        generations=args.generations
    )

if __name__ == "__main__":
    main()