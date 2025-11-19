#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Final NEAT Version: v14 (Fixes Checkpoint Directory Missing Error)

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
import gymnasium as gym
import cv2
from tqdm import tqdm
import pickle
import neat

# [IMPORT] 导入本地 visualization
try:
    import visualize 
    VISUALIZE_ENABLED = True
except ImportError as e:
    print(f"[WARN] Visualization disabled: {e}")
    print("       Ensure 'visualize.py' is in the same directory and 'graphviz' is installed.")
    VISUALIZE_ENABLED = False
except FileNotFoundError as e:
    print(f"[WARN] Graphviz disabled: {e}")
    VISUALIZE_ENABLED = False

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
# 1. Reproducibility & Utils
# ----------------------------
def set_global_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    try: os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    except Exception: pass

class SeedAgeScheduler:
    def __init__(self, seed_pool: Sequence[int], rng_seed: int = 0):
        self.seed_pool = list(seed_pool); self.ages: List[int] = [0 for _ in self.seed_pool]; self.rng = random.Random(rng_seed)
    @classmethod
    def from_fixed_pool(cls, pool_size: int, base_seed: int, shuffle: bool, rng_seed: int):
        g = np.random.default_rng(base_seed); pool = g.integers(low=0, high=2**31-1, size=pool_size, dtype=np.int32).tolist()
        if shuffle: rr = random.Random(rng_seed); rr.shuffle(pool)
        return cls(pool, rng_seed=rng_seed)
    def select_subset(self, k: int) -> List[int]:
        n = len(self.seed_pool); idx = list(range(n)); self.rng.shuffle(idx); idx.sort(key=lambda i: self.ages[i], reverse=True); return [self.seed_pool[i] for i in idx[:k]]
    def update_after_generation(self, chosen_seeds: Sequence[int]):
        chosen = set(chosen_seeds); 
        for i, s in enumerate(self.seed_pool): self.ages[i] = 0 if s in chosen else self.ages[i] + 1
    def get_age_of_seed(self, seed: int) -> int:
        try: return self.ages[self.seed_pool.index(seed)]
        except ValueError: return -1
    def state_dict(self) -> dict: return {"seed_pool": self.seed_pool, "ages": self.ages}
    def load_state_dict(self, d: dict): self.ages = list(d["ages"])

class SeedPortfolioManager:
    def __init__(self, pool_size, base_seed, shuffle, pool_rng_seed, subset_k):
        self.scheduler = SeedAgeScheduler.from_fixed_pool(pool_size, base_seed, shuffle, pool_rng_seed)
        self.master_pool = self.scheduler.seed_pool; self.active_subset = self.scheduler.select_subset(subset_k)
        print(f"🌱 Initial seed subset: {self.active_subset}")
    def get_active_subset(self) -> List[int]: return self.active_subset
    def update_and_refresh(self, results_matrix: np.ndarray, refresh_frac: float, refresh_direction: str, success_threshold: float, max_seed_age: int):
        self.scheduler.update_after_generation(self.active_subset); indices_to_replace = set()
        if max_seed_age > 0:
            age_indices_to_replace = {i for i, s in enumerate(self.active_subset) if self.scheduler.get_age_of_seed(s) > max_seed_age}
            if age_indices_to_replace: indices_to_replace.update(age_indices_to_replace)
        num_perf_replace = int(np.floor(len(self.active_subset) * refresh_frac))
        if num_perf_replace > 0 and refresh_direction != 'none':
            success_matrix = results_matrix > success_threshold; solvers_per_seed = success_matrix.sum(axis=0)
            seed_difficulty_scores = 1.0 / (solvers_per_seed + 1e-9); sorted_indices_by_difficulty = np.argsort(seed_difficulty_scores)
            perf_indices_to_replace = set()
            if refresh_direction == 'bottom': perf_indices_to_replace.update(sorted_indices_by_difficulty[-num_perf_replace:])
            elif refresh_direction == 'top': perf_indices_to_replace.update(sorted_indices_by_difficulty[:num_perf_replace])
            indices_to_replace.update(perf_indices_to_replace)
        if not indices_to_replace: return
        num_to_replace_total = len(indices_to_replace)
        old_seeds_to_replace = {self.active_subset[i] for i in indices_to_replace}
        candidate_pool = [s for s in self.master_pool if s not in self.active_subset]
        if len(candidate_pool) < num_to_replace_total:
            num_to_replace_total = len(candidate_pool); indices_to_replace = list(indices_to_replace)[:num_to_replace_total]
        if num_to_replace_total == 0: return
        new_seeds = random.sample(candidate_pool, num_to_replace_total)
        for i_replace, i_new in zip(indices_to_replace, range(num_to_replace_total)): self.active_subset[i_replace] = new_seeds[i_new]
        print(f"  - Replaced: {list(old_seeds_to_replace)} -> {new_seeds}")
    def state_dict(self): return self.scheduler.state_dict()
    def load_state_dict(self, d): self.scheduler.load_state_dict(d)

# --- 2. RPC & Evaluation ---
class RPCClient:
    def __init__(self, host, port, authkey="yolo-rpc"):
        self.address = (host, port); self.authkey = authkey.encode("utf-8"); self.conn: Optional[Client] = None
    def __enter__(self): self.conn = Client(self.address, authkey=self.authkey); return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn: self.conn.close()
    def reset(self): self.conn.send(("reset", None)); _ = self.conn.recv()
    def update_full(self, frame_bgr, steps=0): self.conn.send(("update_full", (frame_bgr, steps))); ok, z = self.conn.recv(); return z if ok else None
    def predict_only(self): self.conn.send(("predict_only", None)); ok, z = self.conn.recv(); return z if ok else None

def evaluate_individual_neat(args):
    genome_id, genome, config, seed_idx, seed, rpc_host, rpc_port, authkey, max_steps = args
    net = neat.nn.FeedForwardNetwork.create(genome, config); total_reward = 0.0
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
                output = net.activate(state); act = int(np.argmax(output))
                obs, reward, done, truncated, info = env.step(act)
                total_reward += reward
                if done or truncated: break
        env.close(); return genome_id, seed_idx, total_reward
    except Exception: return genome_id, seed_idx, -1000.0

def calculate_competitive_fitness(results_matrix: np.ndarray, success_threshold: float = 0.0) -> np.ndarray:
    population_size, num_seeds = results_matrix.shape
    success_matrix = results_matrix > success_threshold
    solvers_per_seed = success_matrix.sum(axis=0)
    seed_weights = 1.0 / (solvers_per_seed + 1e-9)
    competitive_scores = np.zeros(population_size)
    for i in range(population_size):
        solved_seeds_indices = np.where(success_matrix[i, :])[0]
        if len(solved_seeds_indices) > 0: competitive_scores[i] = seed_weights[solved_seeds_indices].sum()
    return competitive_scores

# --- 6. Logging & Plotting ---
def save_best_genome(run_dir: str, genome: neat.DefaultGenome, config: neat.Config, suffix: str = ""):
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, f"best_genome{suffix}.pkl")
    with open(model_path, 'wb') as f: pickle.dump(genome, f)
    if VISUALIZE_ENABLED:
        try:
            node_names = {-1:'x', -2:'y', -3:'theta', -4:'vx', -5:'vy', -6:'v_theta', -7:'speed', 0:'No-Op', 1:'Left', 2:'Main', 3:'Right'}
            gv_path = os.path.join(run_dir, f"best_genome_net{suffix}.gv")
            visualize.draw_net(config, genome, view=False, filename=gv_path, node_names=node_names, show_disabled=True)
            img_path = os.path.join(run_dir, f"best_genome_net{suffix}")
            try:
                import graphviz
                graph = graphviz.Source.from_file(gv_path)
                graph.render(filename=img_path, format='png', cleanup=True)
            except Exception: pass 
            print(f"  ...Network visualization saved.")
        except Exception as e: print(f"  ...[WARN] Visualize failed: {e}")

def save_metrics_csv(run_dir: str, gen: int, best_raw: float, avg_raw: float, worst_raw: float,
                     best_fit: float, avg_fit: float,
                     avg_nodes: float, min_nodes: int, max_nodes: int):
    path = os.path.join(run_dir, "metrics.csv")
    header_needed = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if header_needed: 
            f.write("generation,best_raw_reward,avg_raw_reward,worst_raw_reward,best_fitness,avg_fitness,avg_nodes,min_nodes,max_nodes\n")
        f.write(f"{gen},{best_raw},{avg_raw},{worst_raw},{best_fit},{avg_fit},{avg_nodes},{min_nodes},{max_nodes}\n")

def plot_training_process_multiview(run_dir: str, gens, fit_best, fit_avg, raw_best, raw_avg, elite_frac):
    if len(gens) == 0: return
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    # View A: Fitness
    ax1.plot(gens, fit_best, label="Best Competitive Fitness", color="purple", linewidth=2)
    ax1.plot(gens, fit_avg, label="Avg Competitive Fitness", color="violet", linestyle="--")
    ax1.set_ylabel("Fitness Score"); ax1.set_title("View A: Selection Pressure (Fitness Sharing)")
    ax1.grid(True, alpha=0.3); ax1.legend()
    # View B: Raw Reward
    ax2.plot(gens, raw_best, label="Best Raw Reward", color="green", linewidth=2)
    ax2.plot(gens, raw_avg, label=f"Elite Top {int(elite_frac*100)}% Avg Raw", color="blue", linestyle="--")
    ax2.set_ylabel("Game Score"); ax2.set_title("View B: Absolute Performance (Raw Reward)")
    ax2.grid(True, alpha=0.3); ax2.legend()
    # View C: Overlay
    ax3.set_title("View C: Correlation Overlay"); ax3.set_xlabel("Generation")
    ax3.set_ylabel("Raw Avg Reward", color="blue")
    line1 = ax3.plot(gens, raw_avg, label="Raw Avg", color="blue", linewidth=2)
    ax3.tick_params(axis='y', labelcolor="blue")
    ax3_twin = ax3.twinx(); ax3_twin.set_ylabel("Fitness Avg", color="purple")
    line2 = ax3_twin.plot(gens, fit_avg, label="Fitness Avg", color="purple", linestyle="--", linewidth=2)
    ax3_twin.tick_params(axis='y', labelcolor="purple")
    lines = line1 + line2; labels = [l.get_label() for l in lines]; ax3.legend(lines, labels, loc='upper left'); ax3.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "training_process_multiview.png"), dpi=150); plt.close()

def plot_complexity_curve(run_dir: str, all_gens, all_avg_nodes, all_min_nodes, all_max_nodes):
    if len(all_gens) == 0: return
    plt.figure(figsize=(10, 6))
    plt.plot(all_gens, all_avg_nodes, label="Average Nodes", color="blue", linewidth=2)
    plt.plot(all_gens, all_max_nodes, label="Max Nodes", color="orange", linestyle="--", alpha=0.8)
    plt.plot(all_gens, all_min_nodes, label="Min Nodes", color="gray", linestyle=":", alpha=0.8)
    plt.fill_between(all_gens, all_min_nodes, all_max_nodes, color='orange', alpha=0.1, label='Range')
    plt.xlabel("Generation"); plt.ylabel("Number of Nodes"); plt.title("Network Complexity Evolution"); plt.grid(True, linestyle="--", alpha=0.4); plt.legend(); out = os.path.join(run_dir, "complexity_curve.png"); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

def plot_final_summary_plots(final_results_matrix: np.ndarray, final_genomes: List[neat.DefaultGenome], master_pool: List[int], run_dir: str):
    try:
        print("📊 Generating final evaluation plots...")
        avg_scores_per_individual = final_results_matrix.mean(axis=1)
        avg_scores_per_seed = final_results_matrix.mean(axis=0)
        final_node_counts = [len(g.nodes) for g in final_genomes]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
        
        # Plot 1: Individual Performance
        sns.violinplot(y=avg_scores_per_individual, ax=ax1, inner='quartile', color='lightblue')
        ax1.set_title(f'Individual Performance\n(N={len(avg_scores_per_individual)})')
        ax1.set_ylabel(f'Avg Reward (256 seeds)')
        ax1.grid(True, linestyle="--", alpha=0.5)
        
        # Plot 2: Seed Difficulty
        sns.violinplot(y=avg_scores_per_seed, ax=ax2, inner='quartile', color='lightgreen')
        ax2.set_title(f'Seed Difficulty\n(N={len(master_pool)})')
        ax2.set_ylabel(f'Avg Reward (120 agents)')
        ax2.grid(True, linestyle="--", alpha=0.5)
        
        # Plot 3: Complexity
        sns.violinplot(y=final_node_counts, ax=ax3, inner='quartile', color='lightcoral')
        ax3.set_title(f'Final Complexity\n(N={len(final_node_counts)})')
        ax3.set_ylabel('Node Count')
        ax3.grid(True, linestyle="--", alpha=0.5)
        
        # [FIX] 手动计算百分比，避免 'normalize' 参数报错
        bins = [-np.inf, 0, 200, np.inf]
        labels = ["Fail (<0)", "Pass (0-200)", "Success (>200)"]
        
        # 显式转换为 Series 以确保兼容性
        seed_scores_series = pd.Series(avg_scores_per_seed)
        categories = pd.cut(seed_scores_series, bins=bins, labels=labels, right=False)
        
        # 获取计数并手动除以总数
        counts = categories.value_counts().sort_index()
        total_seeds = len(avg_scores_per_seed)
        proportions = (counts / total_seeds) * 100
        
        prop_text = "Seed Stats:\n" + "\n".join([f"- {n}: {p:.1f}%" for n, p in proportions.items()])
        fig.text(0.5, 0.01, prop_text, ha='center', fontsize=10, wrap=True)
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        save_path = os.path.join(run_dir, "final_evaluation_plots.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"📈 Final summary plots saved to {save_path}")
        
    except Exception as e:
        print(f"[WARN] Final plots failed: {e}")

# ----------------------------
# 7. Main NEAT Loop
# ----------------------------
def run_neat(processes, rpc_host, rpc_port, authkey, outdir, tag,
           pool_size, subset_k, base_seed, shuffle_pool, pool_rng_seed, global_seed,
           resume, max_steps, success_threshold, seed_refresh_frac, seed_refresh_direction,
           max_seed_age, config_file, generations, fitness_avg_generations):

    if global_seed is not None: set_global_seed(global_seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S"); run_name = f"neat_{ts}" if not tag else f"neat_{ts}_{tag}"
    run_dir = os.path.join(outdir, run_name); os.makedirs(run_dir, exist_ok=True)
    global config; config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    cfg = {k:v for k,v in locals().items() if k not in ['config']}; 
    with open(os.path.join(run_dir, "config.json"), "w") as f: json.dump(cfg, f, indent=2)

    portfolio = SeedPortfolioManager(pool_size, base_seed, shuffle_pool, pool_rng_seed, subset_k)
    seed_age_path = os.path.join(run_dir, "seed_ages.json")
    if resume and os.path.exists(seed_age_path):
        try:
            with open(seed_age_path, "r") as f: portfolio.load_state_dict(json.load(f))
            print(f"🔁 [RESUME] Loaded seed ages")
        except Exception as e: print(f"[WARN] Load seed ages failed: {e}")

    best_global_reward = -np.inf
    # Log lists
    curve_gens = []
    curve_raw_best, curve_raw_avg = [], [] 
    curve_fit_best, curve_fit_avg = [], [] 
    curve_avg_nodes, curve_min_nodes, curve_max_nodes = [], [], []
    
    fitness_history: Dict[int, List[float]] = {} 

    def eval_genomes(genomes, config):
        nonlocal fitness_history
        subset_seeds = portfolio.get_active_subset()
        jobs = []
        genome_id_to_pop_idx_map = {}; pop_idx = 0
        genome_list = list(genomes) 
        
        for genome_id, genome in genome_list:
            genome.fitness = -np.inf; genome_id_to_pop_idx_map[genome_id] = pop_idx
            for seed_idx, seed in enumerate(subset_seeds):
                jobs.append((genome_id, genome, config, seed_idx, seed, rpc_host, rpc_port, authkey, max_steps))
            pop_idx += 1
            
        results_matrix = np.full((len(genome_list), subset_k), -np.inf)
        try:
            with mp.Pool(processes=processes) as pool:
                for genome_id, seed_idx, reward in tqdm(pool.imap_unordered(evaluate_individual_neat, jobs), total=len(jobs), desc=f"Gen {p.generation+1}"):
                    pop_idx = genome_id_to_pop_idx_map[genome_id]; results_matrix[pop_idx, seed_idx] = reward
        except Exception as e: print(f"\n[FATAL] Pool failed: {e}"); raise

        # 1. Calculate Competitive Fitness
        current_competitive_scores = calculate_competitive_fitness(results_matrix, success_threshold)
        
        # 2. Smoothing Logic (Fitness)
        smoothed_scores = np.zeros(len(genome_list)); new_history = {}
        for i, (genome_id, genome) in enumerate(genome_list):
            past = fitness_history.get(genome_id, [])
            hist = (past + [current_competitive_scores[i]])[-fitness_avg_generations:]
            smoothed_scores[i] = np.mean(hist); new_history[genome_id] = hist
            genome.fitness = smoothed_scores[i] # Assign smoothed fitness to NEAT
        fitness_history = new_history

        # 3. Update Seeds
        portfolio.update_and_refresh(results_matrix, seed_refresh_frac, seed_refresh_direction, success_threshold, max_seed_age)
        with open(seed_age_path, "w") as f: json.dump(portfolio.state_dict(), f)
        
        # 4. Statistics for Logging
        gen = p.generation + 1
        raw_avg_rewards = results_matrix.mean(axis=1)
        
        # Fitness Stats (Smoothed)
        fit_best = np.max(smoothed_scores)
        fit_avg = np.mean(smoothed_scores)

        # Raw Reward Stats
        survival_frac = config.reproduction_config.survival_threshold
        elite_indices_fit = np.argsort(smoothed_scores)[-int(survival_frac * len(genome_list)):]
        
        raw_best = float(raw_avg_rewards[np.argmax(smoothed_scores)]) # Reward of the fitness winner
        raw_avg_elite = float(raw_avg_rewards[elite_indices_fit].mean()) # Avg reward of the fitness elites
        raw_worst = float(np.min(raw_avg_rewards))

        # Complexity Stats
        node_counts = [len(g.nodes) for gid, g in genome_list]
        avg_n = np.mean(node_counts); min_n = int(np.min(node_counts)); max_n = int(np.max(node_counts))

        print(f"\n🏆 [GEN {gen:03d}] Fit(Winner)={fit_best:.4f} | Raw(Winner)={raw_best:+.2f} Raw(EliteAvg)={raw_avg_elite:+.2f}")
        print(f"  🧬 Nodes: Avg={avg_n:.1f} Max={max_n} | Species: {len(p.species.species)}")

        # 5. Record & Plot
        curve_gens.append(gen)
        curve_fit_best.append(fit_best); curve_fit_avg.append(fit_avg)
        curve_raw_best.append(raw_best); curve_raw_avg.append(raw_avg_elite)
        curve_avg_nodes.append(avg_n); curve_min_nodes.append(min_n); curve_max_nodes.append(max_n)

        save_metrics_csv(run_dir, gen, raw_best, raw_avg_elite, raw_worst, fit_best, fit_avg, avg_n, min_n, max_n)
        
        plot_training_process_multiview(run_dir, curve_gens, curve_fit_best, curve_fit_avg, curve_raw_best, curve_raw_avg, survival_frac)
        plot_complexity_curve(run_dir, curve_gens, curve_avg_nodes, curve_min_nodes, curve_max_nodes)

        nonlocal best_global_reward
        if raw_best > best_global_reward:
            best_global_reward = raw_best
            best_genome = genome_list[np.argmax(smoothed_scores)][1]
            save_best_genome(run_dir, best_genome, config, suffix="_training_best")
            print(f"💾 [SAVE] New best genome (Raw={best_global_reward:.2f})")

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    
    # [FIX] Explicitly create checkpoints directory
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    p.add_reporter(neat.Checkpointer(5, filename_prefix=os.path.join(checkpoint_dir, 'neat-cp-')))
    
    print(f"🚀 [INFO] Starting NEAT evolution for {generations} generations...")
    winner = p.run(eval_genomes, generations)

    print("\n✅ Training finished.")
    save_best_genome(run_dir, winner, config, suffix="_final_winner")

    # --- Final Eval ---
    print(f"\n🏁 Final evaluation on full {pool_size}-seed pool...")
    master_pool = portfolio.master_pool
    final_genomes = list(p.population.values())
    final_jobs = [(g.key, g, config, j, s, rpc_host, rpc_port, authkey, max_steps) for g in final_genomes for j, s in enumerate(master_pool)]
    
    final_res = np.full((len(final_genomes), len(master_pool)), -np.inf)
    gid_map = {g.key: i for i, g in enumerate(final_genomes)}
    
    try:
        with mp.Pool(processes=processes) as pool:
            for gid, s_idx, rew in tqdm(pool.imap_unordered(evaluate_individual_neat, final_jobs), total=len(final_jobs)):
                final_res[gid_map[gid], s_idx] = rew
    except Exception as e: print(f"[FATAL] Final eval failed: {e}")

    final_avgs = final_res.mean(axis=1)
    best_idx = np.argmax(final_avgs)
    print(f"🏆 [Final] Best Model Avg: {final_avgs[best_idx]:+.2f}")
    save_best_genome(run_dir, final_genomes[best_idx], config, suffix="_full_eval")

    print(f"📝 Saving detailed report...")
    detailed = []
    for i in np.argsort(final_avgs)[::-1][:max(1, int(config.reproduction_config.survival_threshold * len(final_genomes)))]:
        g = final_genomes[i]
        per_seed = {str(master_pool[j]): float(final_res[i, j]) for j in range(len(master_pool))}
        struct = {'nodes': {n: (nd.bias, nd.response, nd.activation, nd.aggregation) for n, nd in g.nodes.items()},
                  'connections': {str(c): (cg.key[0], cg.key[1], cg.weight, cg.enabled) for c, cg in g.connections.items()}}
        detailed.append({"rank": int(i), "id": int(g.key), "avg": float(final_avgs[i]), "structure": struct, "seeds": per_seed})
    
    with open(os.path.join(run_dir, "final_report_detailed.json"), "w") as f: json.dump(detailed, f)
    plot_final_summary_plots(final_res, final_genomes, master_pool, run_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpc-host", default="127.0.0.1"); parser.add_argument("--rpc-port", type=int, default=6001)
    parser.add_argument("--authkey", default="yolo-rpc"); parser.add_argument("--neat-config-file", default="neat.cfg")
    parser.add_argument("--generations", type=int, default=150); parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=256); parser.add_argument("--subset-k", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=202510); parser.add_argument("--shuffle-pool", action="store_true", default=True)
    parser.add_argument("--no-shuffle-pool", dest="shuffle_pool", action="store_false"); parser.add_argument("--pool-rng-seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1000); parser.add_argument("--seed-refresh-frac", type=float, default=0.4)
    parser.add_argument("--seed-refresh-direction", default="bottom", choices=['none', 'top', 'bottom']); parser.add_argument("--max-seed-age", type=int, default=10)
    parser.add_argument("--success-threshold", type=float, default=0.0); parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--outdir", default="runs_neat_final"); parser.add_argument("--tag", default=""); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fitness-avg-generations", type=int, default=5)
    args = parser.parse_args()

    print(f"🚀 [INFO] NEAT Training (v14) | Pop: {args.pool_size} | Refresh: {args.seed_refresh_frac} ({args.seed_refresh_direction})")
    run_neat(args.processes, args.rpc_host, args.rpc_port, args.authkey, args.outdir, args.tag,
             args.pool_size, args.subset_k, args.base_seed, args.shuffle_pool, args.pool_rng_seed, args.global_seed,
             args.resume, args.max_steps, args.success_threshold, args.seed_refresh_frac, args.seed_refresh_direction,
             args.max_seed_age, args.neat_config_file, args.generations, args.fitness_avg_generations)

if __name__ == "__main__":
    main()