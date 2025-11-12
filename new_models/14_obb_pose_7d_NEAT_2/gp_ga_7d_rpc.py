#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Final NEAT Version: v2 (Added Complexity Logging & Plots)

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
import pickle

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
    # torch.manual_seed(seed) # NEAT 主要不依赖 torch 的随机性
    # if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    try:
        # torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    except Exception: pass

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
                indices = sorted_indices_by_difficulty[:num_perf_replace:]; perf_indices_to_replace.update(indices)
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
# 4. NEAT Evaluation Function (Worker)
# ----------------------------
def evaluate_individual_neat(args):
    genome_id, genome, config, seed_idx, seed, rpc_host, rpc_port, authkey, max_steps = args
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
                output = net.activate(state)
                act = int(np.argmax(output))
                obs, reward, done, truncated, info = env.step(act)
                total_reward += reward
                if done or truncated: break
        env.close()
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
def save_best_genome(run_dir: str, genome: neat.DefaultGenome, config: neat.Config, suffix: str = ""):
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, f"best_genome{suffix}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(genome, f)
    try:
        # [MODIFIED] 修正了导入路径
        import neat.visualize
        node_names = {-1:'x', -2:'y', -3:'theta', -4:'vx', -5:'vy', -6:'v_theta', -7:'speed',
                       0:'No-Op', 1:'Left', 2:'Main', 3:'Right'}
        neat.visualize.draw_net(config, genome, view=False, filename=os.path.join(run_dir, f"best_genome_net{suffix}.gv"),
                 node_names=node_names, show_disabled=True)
        print(f"  ...Network visualization saved to .gv file.")
    except Exception as e:
        print(f"  ...[WARN] Could not visualize network (is graphviz installed properly?): {e}")

def save_metrics_csv(run_dir: str, gen: int, best: float, avg_top: float, worst: float,
                     avg_nodes: float, min_nodes: int, max_nodes: int):
    path = os.path.join(run_dir, "metrics.csv")
    header_needed = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if header_needed: 
            f.write("generation,best_avg_reward,avg_top_reward,worst_avg_reward,avg_nodes,min_nodes,max_nodes\n")
        f.write(f"{gen},{best},{avg_top},{worst},{avg_nodes},{min_nodes},{max_nodes}\n")

def plot_reward_curve(run_dir: str, all_gens, all_best, all_avg, all_worst, elite_frac: float):
    if len(all_gens) == 0: return
    plt.figure(figsize=(10, 6)); plt.plot(all_gens, all_best, label="Best Individual Avg Reward", color="green", linewidth=2); plt.plot(all_gens, all_avg, label=f"Top {int(elite_frac*100)}% Avg Reward", color="blue", linestyle="--", alpha=0.8); plt.plot(all_gens, all_worst, label="Worst Individual Avg Reward", color="red", linestyle=":", alpha=0.6); plt.xlabel("Generation"); plt.ylabel("Average Reward over Test Seeds"); plt.title("GA Training Curve"); plt.grid(True, linestyle="--", alpha=0.4); plt.legend(); out = os.path.join(run_dir, "reward_curve.png"); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

def plot_complexity_curve(run_dir: str, all_gens, all_avg_nodes, all_min_nodes, all_max_nodes):
    if len(all_gens) == 0: return
    plt.figure(figsize=(10, 6))
    plt.plot(all_gens, all_avg_nodes, label="Average Nodes", color="blue", linewidth=2)
    plt.plot(all_gens, all_max_nodes, label="Max Nodes", color="orange", linestyle="--", alpha=0.8)
    plt.plot(all_gens, all_min_nodes, label="Min Nodes", color="gray", linestyle=":", alpha=0.8)
    plt.fill_between(all_gens, all_min_nodes, all_max_nodes, color='orange', alpha=0.1, label='Min-Max Range')
    plt.xlabel("Generation"); plt.ylabel("Number of Nodes"); plt.title("NEAT Network Complexity Over Generations"); plt.grid(True, linestyle="--", alpha=0.4); plt.legend(); out = os.path.join(run_dir, "complexity_curve.png"); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

def plot_final_summary_plots(final_results_matrix: np.ndarray, final_genomes: List[neat.DefaultGenome], master_pool: List[int], run_dir: str):
    try:
        print("📊 Generating final evaluation plots...")
        avg_scores_per_individual = final_results_matrix.mean(axis=1)
        avg_scores_per_seed = final_results_matrix.mean(axis=0)
        final_node_counts = [len(g.nodes) for g in final_genomes]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
        sns.violinplot(y=avg_scores_per_individual, ax=ax1, inner='quartile', color='lightblue'); ax1.set_title(f'Distribution of Individual Performance\n(N={len(avg_scores_per_individual)} Individuals)'); ax1.set_ylabel(f'Average Reward (over {len(master_pool)} seeds)'); ax1.grid(True, linestyle="--", alpha=0.5)
        sns.violinplot(y=avg_scores_per_seed, ax=ax2, inner='quartile', color='lightgreen'); ax2.set_title(f'Distribution of Seed Difficulty\n(N={len(master_pool)} Seeds)'); ax2.set_ylabel(f'Average Reward (across {len(avg_scores_per_individual)} individuals)'); ax2.grid(True, linestyle="--", alpha=0.5)
        sns.violinplot(y=final_node_counts, ax=ax3, inner='quartile', color='lightcoral'); ax3.set_title(f'Distribution of Final Network Complexity\n(N={len(final_node_counts)} Individuals)'); ax3.set_ylabel('Node Count'); ax3.grid(True, linestyle="--", alpha=0.5)
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
# 7. Main NEAT Loop
# ----------------------------
def run_neat(processes,
           rpc_host, rpc_port, authkey,
           outdir: str, tag: Optional[str],
           pool_size: int, subset_k: int, base_seed: int, 
           shuffle_pool: bool, pool_rng_seed: int, global_seed: Optional[int],
           resume: bool, max_steps: int, success_threshold: float,
           seed_refresh_frac: float, seed_refresh_direction: str,
           max_seed_age: int,
           config_file: str, generations: int):

    if global_seed is not None: set_global_seed(global_seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S"); run_name = f"neat_{ts}" if not tag else f"neat_{ts}_{tag}"
    run_dir = os.path.join(outdir, run_name); os.makedirs(run_dir, exist_ok=True)
    
    global config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    cfg = {k:v for k,v in locals().items() if k not in ['config']}
    # cfg['neat_config'] = {s: dict(config.genome_config.neat_config[s]) for s in config.genome_config.neat_config.sections()} # This line might be problematic, simplified
    with open(os.path.join(run_dir, "config.json"), "w") as f: json.dump(cfg, f, indent=2)

    portfolio = SeedPortfolioManager(pool_size, base_seed, shuffle_pool, pool_rng_seed, subset_k)
    seed_age_path = os.path.join(run_dir, "seed_ages.json")
    if resume and os.path.exists(seed_age_path):
        try:
            with open(seed_age_path, "r") as f: portfolio.load_state_dict(json.load(f))
            print(f"🔁 [RESUME] Loaded seed ages from {seed_age_path}")
        except Exception as e:
            print(f"[WARN] Failed to load seed ages: {e}.")

    best_global_reward = -np.inf
    curve_gens, curve_best, curve_avg, curve_worst = [], [], [], []
    curve_avg_nodes, curve_min_nodes, curve_max_nodes = [], [], []
    
    def eval_genomes(genomes, config):
        subset_seeds = portfolio.get_active_subset()
        jobs = []
        genome_id_to_pop_idx_map = {}
        pop_idx = 0
        for genome_id, genome in genomes:
            genome.fitness = -np.inf
            genome_id_to_pop_idx_map[genome_id] = pop_idx
            for seed_idx, seed in enumerate(subset_seeds):
                jobs.append((genome_id, genome, config, seed_idx, seed, rpc_host, rpc_port, authkey, max_steps))
            pop_idx += 1
            
        results_matrix = np.full((len(genomes), subset_k), -np.inf)

        try:
            with mp.Pool(processes=processes) as pool:
                results_iterator = tqdm(pool.imap_unordered(evaluate_individual_neat, jobs), total=len(jobs), desc=f"Generation {p.generation+1}/{generations}")
                for genome_id, seed_idx, reward in results_iterator:
                    pop_idx = genome_id_to_pop_idx_map[genome_id]
                    results_matrix[pop_idx, seed_idx] = reward
        except Exception as e:
            print(f"\n[FATAL] Multiprocessing pool failed: {e}"); raise

        competitive_scores = calculate_competitive_fitness(results_matrix, success_threshold)

        for i, (genome_id, genome) in enumerate(genomes):
            genome.fitness = competitive_scores[i]

        portfolio.update_and_refresh(results_matrix, seed_refresh_frac, seed_refresh_direction, success_threshold, max_seed_age)
        with open(seed_age_path, "w") as f: json.dump(portfolio.state_dict(), f)
        
        gen = p.generation + 1
        raw_avg_rewards = results_matrix.mean(axis=1)
        best_genome_idx_by_fitness = np.argmax(competitive_scores)
        best_individual_idx = best_genome_idx_by_fitness
        best_reward = float(raw_avg_rewards[best_individual_idx])
        
        # [MODIFIED] 从 config 中读取 survival_threshold
        survival_frac = config.reproduction_config.survival_threshold 
        elite_num = max(2, int(survival_frac * len(genomes)))
        elite_indices = np.argsort(competitive_scores)[-elite_num:]
        avg_top_reward = float(raw_avg_rewards[elite_indices].mean())
        
        worst_reward = float(np.min(raw_avg_rewards))
        
        node_counts = [len(g.nodes) for gid, g in genomes]
        avg_nodes = np.mean(node_counts)
        min_nodes = int(np.min(node_counts))
        max_nodes = int(np.max(node_counts))
        
        print(f"\n🏆 [GEN {gen:03d}] BestAvgReward(Comp.Winner)={best_reward:+.2f}  AvgTop{elite_num}(Comp.Elite)={avg_top_reward:+.2f}  WorstAvgReward={worst_reward:+.2f}")
        print(f"  🧬 Complexity: Min={min_nodes} Avg={avg_nodes:.1f} Max={max_nodes}  Species: {len(p.species.species)}")

        curve_gens.append(gen); curve_best.append(best_reward); curve_avg.append(avg_top_reward); curve_worst.append(worst_reward)
        curve_avg_nodes.append(avg_nodes); curve_min_nodes.append(min_nodes); curve_max_nodes.append(max_nodes)
        
        save_metrics_csv(run_dir, gen, best_reward, avg_top_reward, worst_reward, avg_nodes, min_nodes, max_nodes)
        plot_reward_curve(run_dir, curve_gens, curve_best, curve_avg, curve_worst, survival_frac) # [MODIFIED] 传入 survival_frac
        plot_complexity_curve(run_dir, curve_gens, curve_avg_nodes, curve_min_nodes, curve_max_nodes)

        nonlocal best_global_reward
        if best_reward > best_global_reward:
            best_global_reward = best_reward
            best_genome_this_gen = genomes[best_individual_idx][1]
            save_best_genome(run_dir, best_genome_this_gen, config, suffix="_training_best")
            print(f"💾 [SAVE] New best training genome to {run_dir} (avg reward {best_global_reward:.2f})")

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    p.add_reporter(neat.Checkpointer(generation_interval=5, filename_prefix=os.path.join(checkpoint_dir, 'neat-cp-')))
    
    print(f"🚀 [INFO] Starting NEAT evolution for {generations} generations...")
    winner = p.run(eval_genomes, generations)

    print("\n✅ Training finished.")
    print(f"💾 [SAVE] Saving final best genome (winner)...")
    save_best_genome(run_dir, winner, config, suffix="_final_winner")

    # ----------------------------
    # 9. Final Full Evaluation
    # ----------------------------
    print(f"\n🏁 Starting final evaluation on full {pool_size}-seed pool...")
    master_pool = portfolio.master_pool
    
    final_pop_genomes = list(p.population.values())
    final_pop_ids = list(p.population.keys())
    
    final_jobs = []
    for i, genome in enumerate(final_pop_genomes):
        genome_id = genome.key
        for j, seed in enumerate(master_pool):
            final_jobs.append((genome_id, genome, config, j, seed, rpc_host, rpc_port, authkey, max_steps))
            
    final_results_matrix = np.full((len(final_pop_genomes), len(master_pool)), -np.inf)
    genome_id_to_pop_idx_map = {gid: i for i, gid in enumerate(final_pop_ids)}
    
    try:
        with mp.Pool(processes=processes) as pool:
            results_iterator = tqdm(pool.imap_unordered(evaluate_individual_neat, final_jobs), total=len(final_jobs), desc="Final Full Evaluation")
            for genome_id, seed_idx, reward in results_iterator:
                pop_idx = genome_id_to_pop_idx_map.get(genome_id) # Use .get for safety
                if pop_idx is not None:
                    final_results_matrix[pop_idx, seed_idx] = reward
    except Exception as e:
        print(f"\n[FATAL] Final evaluation failed: {e}")

    final_avg_scores = final_results_matrix.mean(axis=1)
    final_sorted_indices = np.argsort(final_avg_scores)[::-1]

    best_final_idx_in_pop = final_sorted_indices[0]
    best_final_score = final_avg_scores[best_final_idx_in_pop]
    best_final_genome = final_pop_genomes[best_final_idx_in_pop]

    print(f"🏆 [Final Result] Best model from full {pool_size}-seed eval: {best_final_score:+.2f} (Genome ID {best_final_genome.key})")
    
    save_best_genome(run_dir, best_final_genome, config, suffix="_full_eval")
    print(f"💾 [SAVE] Best model (full eval) saved to {run_dir}")
    
    print(f"📝 Saving detailed final report...")
    detailed_report = []
    survival_frac_from_cfg = config.reproduction_config.survival_threshold
    num_to_report = max(1, int(survival_frac_from_cfg * len(final_pop_genomes)))
    
    for rank, pop_idx in enumerate(final_sorted_indices[:num_to_report]):
        genome = final_pop_genomes[pop_idx]
        ind_scores = final_results_matrix[pop_idx, :]
        per_seed_results = {str(master_pool[j]): float(ind_scores[j]) for j in range(len(master_pool))}
        
        # [MODIFIED] 修复了 AttributeError
        genome_struct = {
            'nodes': {nid: (n.bias, n.response, n.activation, n.aggregation) for nid, n in genome.nodes.items()},
            'connections': {
                cid: (c.key[0], c.key[1], c.weight, c.enabled) 
                for cid, c in genome.connections.items()
            }
        }
        
        detailed_report.append({
            "final_rank": rank + 1,
            "genome_id": genome.key,
            "avg_reward_full": float(final_avg_scores[pop_idx]),
            "genome_structure": genome_struct,
            "per_seed_rewards": per_seed_results
        })
    
    with open(os.path.join(run_dir, "final_report_detailed.json"), "w") as f:
        json.dump(detailed_report, f)
    print(f"💾 [SAVE] Detailed final report for top {num_to_report} individuals saved to {run_dir}")
    
    plot_final_summary_plots(final_results_matrix, final_pop_genomes, master_pool, run_dir)

# ----------------------------
# 10. CLI (Command Line Interface)
# ----------------------------
def main():
    global args # [MODIFIED] 设为全局，以便绘图函数能访问
    parser = argparse.ArgumentParser(description="NEAT Client with Competitive Fitness and Dynamic Seed Refresh")
    # RPC
    parser.add_argument("--rpc-host", type=str, default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=6001)
    parser.add_argument("--authkey", type=str, default="yolo-rpc")
    
    # NEAT Core
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

    # [MODIFIED] 修复了 TypeError
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
        config_file=args.neat_config_file,
        generations=args.generations
    )

if __name__ == "__main__":
    main()