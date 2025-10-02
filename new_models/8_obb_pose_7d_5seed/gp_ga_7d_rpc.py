#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# gp_ga_7d_rpc.py
import os
import argparse
import random
import json
import multiprocessing as mp
from multiprocessing.connection import Client
from typing import Optional, Tuple, Sequence, List
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
# [FIX] 确保导入 cv2
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# 为多进程设置 PyTorch 线程数，避免资源竞争
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 让 tqdm 输出更干净
print = lambda *a, **k: (__import__("builtins").print(*a, **{**k, "flush": True}))

# ----------------------------
# 复现实验：全局随机种子
# ----------------------------
def set_global_seed(seed: int):
    """
    固定 Python / NumPy / Torch 随机源，并尽量启用确定性。
    注意：多进程执行顺序仍可能带来微小非确定性；但同一输入将得到相同评估结果。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 尽量确定性（可能稍降速）
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    except Exception:
        pass

# ----------------------------
# 简单 2层 MLP: 输入 7D -> 输出 4 动作 logits
# ----------------------------
class NNPolicy(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden), nn.ReLU(),
            nn.Linear(hidden, 4)
        )

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
# 64 固定 seed + 动态子集（按年龄优先）
# ----------------------------
class SeedAgeScheduler:
    """
    - 维护固定 seed 池与年龄（age = 自上次被选中后的代数）。
    - 每代：select_subset(k) 取年龄最大的 k 个 seed；
      评估完成后 update_after_generation(chosen_seeds)：
        - 选中的 age 置 0；
        - 未选中的 age += 1。
    """
    def __init__(self, seed_pool: Sequence[int], rng_seed: int = 0):
        self.seed_pool = list(seed_pool)
        self.ages: List[int] = [0 for _ in self.seed_pool]
        self.rng = random.Random(rng_seed)

    @classmethod
    def from_fixed_pool(cls, pool_size: int = 64, base_seed: int = 12345, shuffle: bool = True, rng_seed: int = 0):
        g = np.random.default_rng(base_seed)
        pool = g.integers(low=0, high=2**31-1, size=pool_size, dtype=np.int32).tolist()
        if shuffle:
            rr = random.Random(rng_seed)
            rr.shuffle(pool)
        return cls(pool, rng_seed=rng_seed)

    def select_subset(self, k: int) -> List[int]:
        n = len(self.seed_pool)
        if k >= n:
            return list(self.seed_pool)
        idx = list(range(n))
        self.rng.shuffle(idx)  # 打散 tie
        idx.sort(key=lambda i: self.ages[i], reverse=True)
        chosen_idx = idx[:k]
        return [self.seed_pool[i] for i in chosen_idx]

    def update_after_generation(self, chosen_seeds: Sequence[int]):
        chosen = set(chosen_seeds)
        for i, s in enumerate(self.seed_pool):
            if s in chosen:
                self.ages[i] = 0
            else:
                self.ages[i] += 1

    # 断点续训
    def state_dict(self) -> dict:
        return {"seed_pool": self.seed_pool, "ages": self.ages}

    def load_state_dict(self, d: dict):
        if d["seed_pool"] != self.seed_pool:
            raise ValueError("Seed pool mismatch: 请确保相同 base_seed/pool_size/shuffle 配置。")
        self.ages = list(d["ages"])

# ----------------------------
# RPC Client for 7D state
# ----------------------------
class RPCClient:
    def __init__(self, host: str, port: int, authkey: str = "yolo-rpc"):
        self.address = (host, port)
        self.authkey = authkey.encode("utf-8")
        self.conn: Optional[Client] = None

    def __enter__(self):
        try:
            self.conn = Client(self.address, authkey=self.authkey)
        except ConnectionRefusedError:
            print(f"[FATAL] Connection to RPC server {self.address} refused. Is the server running?")
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def reset(self):
        self.conn.send(("reset", None))
        _ = self.conn.recv()

    # [FIX] 将参数名改为 steps，更清晰
    def update_full(self, frame_bgr: np.ndarray, steps: int = 0):
        self.conn.send(("update_full", (frame_bgr, steps)))
        ok, z = self.conn.recv()
        return z if ok else None

    def predict_only(self):
        self.conn.send(("predict_only", None))
        ok, z = self.conn.recv()
        return z if ok else None

# ----------------------------
# 个体评估：在给定的 seed 子集上取平均 reward
# ----------------------------
def rollout_one_seed(model: NNPolicy, rpc_host: str, rpc_port: int, authkey: str, seed: int, max_steps: int = 1000) -> float:
    """在单个 Gym 种子上跑一回合，返回reward"""
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0

    try:
        with RPCClient(rpc_host, rpc_port, authkey) as rpc:
            rpc.reset()  # 每个 episode 重置 7D 估计器状态

            for t in range(max_steps):
                frame = env.render()
                if frame is None:
                    break
                # RGB -> BGR
                frame_bgr = frame[..., ::-1]
                state = rpc.update_full(frame_bgr, t)
                if state is None:
                    state = rpc.predict_only()
                if state is None:
                    state = np.zeros(7, dtype=np.float32)

                s = torch.tensor(state, dtype=torch.float32)
                with torch.no_grad():
                    act = int(torch.argmax(model(s)).item())

                obs, reward, done, truncated, info = env.step(act)
                total_reward += reward
                if done or truncated:
                    break
    except Exception:
        total_reward = -1000.0
    finally:
        env.close()

    return float(total_reward)

def evaluate_individual(args):
    """
    多进程 worker：对一个个体在 subset_seeds 上评估平均 reward
    """
    idx, weights, rpc_host, rpc_port, authkey, subset_seeds, max_steps = args
    model = NNPolicy()
    set_weights_vector(model, weights)

    rewards = []
    for sd in subset_seeds:
        r = rollout_one_seed(model, rpc_host, rpc_port, authkey, seed=sd, max_steps=max_steps)
        rewards.append(r)

    avg_reward = float(np.mean(rewards)) if len(rewards) > 0 else -1000.0
    return idx, avg_reward

# ----------------------------
# GA 主循环
# ----------------------------
def run_ga(population_size, generations, elite_frac, sigma, processes,
           rpc_host, rpc_port, authkey,
           # 新增：动态子集 & 复现 & 输出
           pool_size, subset_k, base_seed, shuffle_pool, pool_rng_seed,
           global_seed, outdir, resume, max_steps):

    # 全局随机种子（可选）
    if global_seed is not None:
        set_global_seed(global_seed)

    os.makedirs(outdir, exist_ok=True)
    seed_age_path = os.path.join(outdir, "seed_ages.json")
    metrics_csv = os.path.join(outdir, "metrics.csv")
    plot_path = os.path.join(outdir, "training_curve.png")
    best_model_path = os.path.join(outdir, "best_model.pt")

    # 初始化 Seed 调度器
    scheduler = SeedAgeScheduler.from_fixed_pool(
        pool_size=pool_size,
        base_seed=base_seed,
        shuffle=shuffle_pool,
        rng_seed=pool_rng_seed
    )
    if resume and os.path.exists(seed_age_path):
        try:
            with open(seed_age_path, "r", encoding="utf-8") as f:
                scheduler.load_state_dict(json.load(f))
            print(f"🔁 [RESUME] Loaded seed ages from {seed_age_path}")
        except Exception as e:
            print(f"[WARN] Failed to load seed ages: {e}. Will start fresh.")

    # 初始化种群
    model = NNPolicy()
    base_vec = get_weights_vector(model)
    pop = [mutate(base_vec, sigma=0.2) for _ in range(population_size)]

    # 记录指标
    gen_list, best_list, top_list, worst_list = [], [], [], []
    best_score_global = -np.inf
    best_weights_global = None

    # CSV 头
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", encoding="utf-8") as f:
            f.write("generation,best,top10_avg,worst\n")

    for gen in range(1, generations + 1):
        # 选择“年龄最大”的子集种子
        subset_seeds = scheduler.select_subset(k=subset_k)

        # 多进程评估
        jobs = [(i, pop[i], rpc_host, rpc_port, authkey, subset_seeds, max_steps) for i in range(population_size)]
        scores = np.zeros(population_size)
        try:
            with mp.Pool(processes=processes) as pool:
                results = list(tqdm(pool.imap_unordered(evaluate_individual, jobs),
                                    total=len(jobs),
                                    desc=f"Generation {gen}/{generations}"))
            for idx, r in results:
                scores[idx] = r
        except Exception as e:
            print(f"\n[FATAL] Multiprocessing pool failed: {e}")
            print("Please ensure the RPC server is running and accessible.")
            break

        # 更新 seed 年龄（被选中置 0，未选中 +1）
        scheduler.update_after_generation(chosen_seeds=subset_seeds)
        # 落盘
        try:
            with open(seed_age_path, "w", encoding="utf-8") as f:
                json.dump(scheduler.state_dict(), f)
        except Exception as e:
            print(f"[WARN] Failed to save seed ages: {e}")

        # 选择精英与统计
        elite_num = max(2, int(elite_frac * population_size))
        elite_idx = np.argsort(scores)[-elite_num:]
        elites = [pop[i] for i in elite_idx]

        best = scores[elite_idx[-1]]
        avg_top = scores[elite_idx].mean()
        worst = float(np.min(scores))
        print(f"🏆 [GEN {gen:03d}] Best={best:+.2f}  AvgTop{elite_num}={avg_top:+.2f}  Worst={worst:+.2f}  Sigma={sigma:.3f}")

        # 记录
        gen_list.append(gen)
        best_list.append(float(best))
        top_list.append(float(avg_top))
        worst_list.append(float(worst))
        # 追加 CSV
        with open(metrics_csv, "a", encoding="utf-8") as f:
            f.write(f"{gen},{best:.6f},{avg_top:.6f},{worst:.6f}\n")

        # 保存全局最佳模型
        if best > best_score_global:
            best_score_global = float(best)
            best_weights_global = pop[int(elite_idx[-1])]
            best_model = NNPolicy()
            set_weights_vector(best_model, best_weights_global)
            torch.save(best_model.state_dict(), best_model_path)
            print(f"💾 [SAVE] New best model saved to {best_model_path} (score {best_score_global:.2f})")

        # 产生下一代
        new_pop = elites.copy()
        while len(new_pop) < population_size:
            p1, p2 = random.sample(elites, 2)
            c1, c2 = uniform_crossover(p1, p2)
            new_pop.append(mutate(c1, sigma))
            if len(new_pop) < population_size:
                new_pop.append(mutate(c2, sigma))
        pop = new_pop

        # 退火
        sigma = max(0.01, sigma * 0.99)

        # 每代画一次最新曲线（覆盖写入）
        try:
            plt.figure(figsize=(8, 5), dpi=120)
            plt.plot(gen_list, best_list, label="Best reward")
            plt.plot(gen_list, top_list, label="Top 10% avg reward")
            plt.plot(gen_list, worst_list, label="Worst reward")
            plt.xlabel("Generation")
            plt.ylabel("Reward")
            plt.title("GA Training Curve")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
        except Exception as e:
            print(f"[WARN] Failed to save plot: {e}")

    print("✅ Training finished.")
    if best_weights_global is not None:
        print(f"🏁 Best score observed: {best_score_global:.2f}")
        print(f"📦 Best model stored at: {best_model_path}")
        print(f"📈 Metrics plot: {plot_path}")
        print(f"🧾 Metrics CSV: {metrics_csv}")

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="GA Client for LunarLander with RPC state estimation")
    # 原有参数
    parser.add_argument("--rpc-host", type=str, default="127.0.0.1", help="RPC server host")
    parser.add_argument("--rpc-port", type=int, default=6001, help="RPC server port")
    parser.add_argument("--authkey", type=str, default="yolo-rpc", help="RPC authentication key")
    parser.add_argument("--processes", type=int, default=8, help="Number of parallel processes")
    parser.add_argument("--population", type=int, default=120, help="Population size")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations")
    parser.add_argument("--elite-frac", type=float, default=0.1, help="Fraction of elites to keep")
    parser.add_argument("--sigma", type=float, default=0.1, help="Initial mutation strength")

    # 新增参数：动态子集 & 复现 & 输出
    parser.add_argument("--pool-size", type=int, default=64, help="Fixed seed pool size (default 64)")
    parser.add_argument("--subset-k", type=int, default=5, help="How many seeds to evaluate per generation")
    parser.add_argument("--base-seed", type=int, default=202509, help="Generator seed to produce the fixed seed pool")
    parser.add_argument("--shuffle-pool", action="store_true", default=True, help="Shuffle the fixed pool")
    parser.add_argument("--no-shuffle-pool", dest="shuffle_pool", action="store_false", help="Do not shuffle seed pool")
    parser.add_argument("--pool-rng-seed", type=int, default=0, help="Tie-breaker RNG seed for same-age seeds")

    parser.add_argument("--global-seed", type=int, default=None, help="Set this to fully fix randomness for reproducibility")
    parser.add_argument("--outdir", type=str, default="runs_ga", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume seed ages from outdir/seed_ages.json if present")

    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    args = parser.parse_args()

    print(f"🚀 [INFO] Starting GA training with {args.processes} processes, "
          f"pop={args.population}, gen={args.generations}")
    print(f"📡 Connecting to RPC server at {args.rpc_host}:{args.rpc_port}")

    run_ga(args.population, args.generations, args.elite_frac, args.sigma,
           args.processes, args.rpc_host, args.rpc_port, args.authkey,
           pool_size=args.pool_size, subset_k=args.subset_k, base_seed=args.base_seed,
           shuffle_pool=args.shuffle_pool, pool_rng_seed=args.pool_rng_seed,
           global_seed=args.global_seed, outdir=args.outdir, resume=args.resume,
           max_steps=args.max_steps)

if __name__ == "__main__":
    # Windows 下 multiprocessing 需要此保护
    main()
