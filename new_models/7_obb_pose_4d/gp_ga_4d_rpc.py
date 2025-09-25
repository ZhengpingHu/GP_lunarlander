#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import time
import argparse
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from tqdm import tqdm

# 避免 OpenMP 多副本告警（仅客户端侧，模型在服务端 GPU）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import gymnasium as gym
import torch
import torch.nn as nn

# =========================================================
# 4D 小网络：输入 4 维 -> 4 动作，简单 MLP
# =========================================================
class NNPolicy(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(),
            nn.Linear(hidden, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def get_weights_vector(model: nn.Module) -> np.ndarray:
    return torch.cat([p.data.flatten() for p in model.parameters()]).cpu().numpy()

def set_weights_vector(model: nn.Module, vec: np.ndarray) -> None:
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(torch.from_numpy(vec[offset:offset+n]).view_as(p))
        offset += n

def uniform_crossover(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.random.rand(p1.size) < 0.5
    c1 = np.where(mask, p1, p2)
    c2 = np.where(~mask, p1, p2)
    return c1, c2

def mutate(vec: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return vec.copy()
    return vec + np.random.randn(vec.size) * sigma

# =========================================================
# YOLO 状态估计的 RPC 客户端（与服务端单进程 GPU 推理配合）
# =========================================================
from multiprocessing.connection import Client

@dataclass
class RemoteEstimator:
    address: Tuple[str, int]
    authkey: bytes = b"yolo-rpc"
    conn: Optional[Client] = None

    def _ensure(self):
        if self.conn is None:
            self.conn = Client(self.address, authkey=self.authkey)

    def ping(self) -> bool:
        try:
            self._ensure()
            self.conn.send(("ping", None))
            ok, _ = self.conn.recv()
            return bool(ok)
        except Exception:
            return False

    def reset(self):
        self._ensure()
        self.conn.send(("reset", None))
        _ = self.conn.recv()

    def update_full(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        self._ensure()
        self.conn.send(("update_full", frame_bgr))
        ok, z = self.conn.recv()
        return z if ok else None

    def predict_only(self) -> Optional[np.ndarray]:
        self._ensure()
        self.conn.send(("predict_only", None))
        ok, z = self.conn.recv()
        return z if ok else None

# =========================================================
# 单回合评估（固定 seed，轻量 frame-skip，全部视觉在服务端）
# =========================================================
def _evaluate_one_episode(model: NNPolicy, rpc: RemoteEstimator, seed: int,
                          max_steps: int = 1000, skip: int = 8) -> float:
    # 注意：你现有环境可能是 v3；若只支持 v2，请改成 LunarLander-v2
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    try:
        obs, _ = env.reset(seed=seed)
        rpc.reset()
        total = 0.0
        done = False

        # 初始感知
        frame = env.render()
        state = rpc.update_full(frame[..., ::-1])    # RGB -> BGR
        if state is None:
            return -1000.0

        steps = 0
        with torch.no_grad():
            while not done and steps < max_steps:
                s = torch.tensor(state, dtype=torch.float32)
                act = int(torch.argmax(model(s)).item())

                obs, r, term, trunc, _info = env.step(act)
                total += r
                done = term or trunc

                if steps % skip == 0:
                    frame = env.render()
                    state = rpc.update_full(frame[..., ::-1])
                    if state is None:
                        return total - 300.0
                else:
                    state = rpc.predict_only()
                    if state is None:
                        return total - 200.0

                steps += 1

        return float(total)
    finally:
        env.close()

# =========================================================
# DSS（Dynamic Subset Selection）——跨代维护种子统计，代内统一子集
# =========================================================
class SeedStats:
    def __init__(self, seeds: List[int], alpha: float = 0.2):
        self.seeds = list(seeds)
        self.alpha = alpha
        self.stats: Dict[int, Dict[str, float]] = {
            s: dict(age=5.0, n=0.0, mean=0.0, var=1.0, solve=0.0) for s in self.seeds
        }
        # 动态解题阈值（fallback）
        self.R_solve: float = 100.0

    def tick_ages(self):
        for st in self.stats.values():
            st["age"] += 1.0

    def _norm(self, arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        a, b = float(arr.min()), float(arr.max())
        if b <= a + 1e-12:
            return np.zeros_like(arr)
        return (arr - a) / (b - a)

    def pick_subset(self, k: int,
                    wA: float = 0.5, wD: float = 0.3, wH: float = 0.2) -> List[int]:
        ages = np.array([self.stats[s]["age"]  for s in self.seeds], dtype=float)
        vars_ = np.array([self.stats[s]["var"]  for s in self.seeds], dtype=float)
        means = np.array([self.stats[s]["mean"] for s in self.seeds], dtype=float)

        A = self._norm(ages)
        D = self._norm(vars_)
        H = 1.0 - self._norm(means)

        scores = wA * A + wD * D + wH * H
        idx = np.argsort(-scores)  # 降序
        top = idx[:max(2*k, k)]

        chosen = set([self.seeds[idx[0]]])  # 至少包含分数最高的一个
        probs = scores[top]
        psum = float(probs.sum()) + 1e-8
        probs = probs / psum

        while len(chosen) < k:
            j = int(np.random.choice(top, p=probs))
            chosen.add(self.seeds[j])

        return list(chosen)

    def update_per_seed(self, s: int, rewards: List[float]):
        if len(rewards) == 0:
            return
        m = float(np.mean(rewards))
        v = float(np.var(rewards))
        solved = float(np.mean([r >= self.R_solve for r in rewards]))

        st = self.stats[s]
        a = self.alpha
        st["mean"]  = (1 - a) * st["mean"]  + a * m
        st["var"]   = (1 - a) * st["var"]   + a * v
        st["solve"] = (1 - a) * st["solve"] + a * solved
        st["n"] += 1.0
        st["age"] = 0.0

    def set_dynamic_solve_threshold(self, top10_median: float):
        # 建议用当代 top10% 中位数的 0.8 做动态达标阈值
        self.R_solve = 0.8 * float(top10_median)

# =========================================================
# 多进程 worker：评估（个体参数 + 指定 seed）
# =========================================================
def _eval_job(args: Tuple[np.ndarray, Tuple[str, int], int]) -> Tuple[int, float]:
    vec, rpc_addr, seed = args
    model = NNPolicy()
    set_weights_vector(model, vec)
    rpc = RemoteEstimator(rpc_addr)
    # 尝试快速 ping，失败也继续（服务端初次加载或瞬断）
    try:
        rpc.ping()
    except Exception:
        pass
    try:
        rew = _evaluate_one_episode(model, rpc, seed)
    except Exception:
        rew = -1000.0
    return (seed, float(rew))

# =========================================================
# GA 主过程：DSS 子集 -> 全体粗评（统一子集）-> TopK 重评（固定基准集）
# =========================================================
def run_ga(population_size: int,
           generations: int,
           keep_ratio: float,
           processes: int,
           rpc_host: str,
           rpc_port: int,
           coarse_subset_k: int,
           reeval_topk: float,
           reeval_seeds: int,
           sigma_init: float = 0.15,
           sigma_min: float = 0.05,
           seed_pool_size: int = 64,
           random_seed_base: int = 1000):

    rng = np.random.default_rng(42)
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    rpc_addr = (rpc_host, int(rpc_port))

    # 初始化一个模型来拿到参数维度
    m = NNPolicy()
    w0 = get_weights_vector(m)
    dim = w0.size

    # 初始种群（高斯）
    pop: List[np.ndarray] = [w0 + 0.05 * np.random.randn(dim) for _ in range(population_size)]

    # DSS 统计与种子池
    pool_seeds = list(range(random_seed_base, random_seed_base + seed_pool_size))
    dss = SeedStats(pool_seeds, alpha=0.2)

    # 固定基准子集（用于重评对比，覆盖易/中/难；可按需改）
    K_BENCH: List[int] = [2001, 2002, 2003, 2004, 2011, 2012, 2021, 2022]

    # 退火
    sigma = float(sigma_init)
    decay = (sigma_min / sigma_init) ** (1.0 / max(1, generations))

    print(f"🚀 [INFO] Starting GA training with {processes} parallel processes.")
    print(f"[INFO] seed_pool={seed_pool_size}  coarse_subset_k={coarse_subset_k}  reeval_topk={reeval_topk}  bench={len(K_BENCH)}  sigma_init={sigma_init}")

    from multiprocessing import Pool
    for gen in range(1, generations + 1):
        t0 = time.time()

        # 1) DSS：更新“年龄”，挑当代统一子集
        dss.tick_ages()
        subset = dss.pick_subset(k=coarse_subset_k)

        # 2) 粗评（统一子集）：每个（个体×子集种子）一个任务
        jobs = []
        for vec in pop:
            for s in subset:
                jobs.append((vec, rpc_addr, s))

        # 收集：对每个个体，取“子集上 reward 的中位数”作为粗评成绩
        # 同时累积每个 seed 的 reward 列表以更新 DSS 统计
        indiv_rewards: List[List[float]] = [[] for _ in range(population_size)]
        seed_to_rewards: Dict[int, List[float]] = {s: [] for s in subset}

        with Pool(processes=processes) as pool:
            results = list(tqdm(pool.imap_unordered(_eval_job, jobs),
                                total=len(jobs), desc=f"Generation {gen}/{generations} - Coarse"))
        # 结果按提交顺序对应不了个体索引，所以再跑一遍映射
        # 我们知道 jobs 的结构：按个体循环，再按子集循环。可以按块回填。
        # 更简单的方法：重跑一次循环计数器：
        idx = 0
        for i in range(population_size):
            for _k in range(len(subset)):
                s, r = results[idx]
                indiv_rewards[i].append(r)
                seed_to_rewards[s].append(r)
                idx += 1

        coarse_scores = np.array([np.median(rs) if len(rs) else -1000.0 for rs in indiv_rewards], dtype=float)

        # 动态达标阈值：用当代 top10% 的中位数 * 0.8
        k10 = max(1, int(0.1 * population_size))
        top10_med = float(np.median(np.sort(coarse_scores)[-k10:]))
        dss.set_dynamic_solve_threshold(top10_med)

        # 回写 seed 统计
        for s in subset:
            dss.update_per_seed(s, seed_to_rewards[s])

        # 3) Top-K 重评（固定基准子集）：对当代前 reeval_topk 的个体
        topk = max(1, int(reeval_topk * population_size))
        top_idx = np.argsort(coarse_scores)[-topk:]
        reeval_jobs = []
        for i in top_idx:
            for s in K_BENCH:
                reeval_jobs.append((pop[i], rpc_addr, s))

        bench_scores = {}
        if len(reeval_jobs) > 0:
            with Pool(processes=processes) as pool:
                reeval_results = list(tqdm(pool.imap_unordered(_eval_job, reeval_jobs),
                                           total=len(reeval_jobs), desc=f"Generation {gen}/{generations} - ReEval"))
            # 聚合每个体的重评结果
            ptr = 0
            for i in top_idx:
                rs = []
                for _k in range(len(K_BENCH)):
                    s, r = reeval_results[ptr]
                    rs.append(r)
                    ptr += 1
                rs = np.array(rs, dtype=float)
                median = float(np.median(rs))
                p90 = float(np.percentile(rs, 90))
                bench_scores[i] = 0.7 * median + 0.3 * p90

        # 最终适应度：有重评用重评，否则用粗评
        fitness = coarse_scores.copy()
        for i, fit in bench_scores.items():
            fitness[i] = fit

        # 打印日志
        best = float(np.max(fitness))
        avg_top10 = float(np.mean(np.sort(fitness)[-k10:]))
        cbest = float(np.max(coarse_scores))
        cavg_top10 = float(np.mean(np.sort(coarse_scores)[-k10:]))
        print(f"[GEN {gen:03d}] best={best:+.2f}  avg-top10%={avg_top10:+.2f}  "
              f"(coarse-best={cbest:+.2f} avg-top10%={cavg_top10:+.2f})  sigma={sigma:.3f}  time={time.time()-t0:.1f}s")

        # 4) 选择 + 产生下一代（精英保留 + 均匀交叉 + 高斯突变）
        keep = max(2, int(keep_ratio * population_size))
        order = np.argsort(fitness)[::-1]
        elites = [pop[i].copy() for i in order[:keep]]
        next_pop: List[np.ndarray] = []
        next_pop.extend(elites)

        # 父本池（从前 50% 中采样父母）
        parent_pool_idx = order[:max(keep, population_size // 2)]
        while len(next_pop) < population_size:
            i, j = np.random.choice(parent_pool_idx, size=2, replace=True)
            c1, c2 = uniform_crossover(pop[i], pop[j])
            next_pop.append(mutate(c1, sigma))
            if len(next_pop) < population_size:
                next_pop.append(mutate(c2, sigma))

        pop = next_pop

        # 退火
        sigma = max(sigma_min, sigma * decay)

# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=200, help="种群规模")
    parser.add_argument("--generations", type=int, default=80, help="迭代代数")
    parser.add_argument("--keep-ratio", type=float, default=0.20, help="精英比例")
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 4) - 2), help="并行进程数")
    parser.add_argument("--rpc-host", type=str, default="127.0.0.1", help="RPC 服务端 host")
    parser.add_argument("--rpc-port", type=int, default=6001, help="RPC 服务端 port")

    # DSS / 评估相关
    parser.add_argument("--seed-pool-size", type=int, default=64, help="固定种子池容量")
    parser.add_argument("--coarse-subset-k", type=int, default=5, help="每代统一粗评子集大小")
    parser.add_argument("--reeval-topk", type=float, default=0.2, help="每代参与重评的前百分比")
    parser.add_argument("--reeval-seeds", type=int, default=8, help="重评使用基准子集大小（建议与固定基准等长）")

    # 退火
    parser.add_argument("--sigma-init", type=float, default=0.15, help="初始突变强度")
    parser.add_argument("--sigma-min", type=float, default=0.05, help="最小突变强度")

    args = parser.parse_args()

    # Windows 建议
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    run_ga(population_size=args.population,
           generations=args.generations,
           keep_ratio=args.keep_ratio,
           processes=args.processes,
           rpc_host=args.rpc_host,
           rpc_port=args.rpc_port,
           coarse_subset_k=args.coarse_subset_k,
           reeval_topk=args.reeval_topk,
           reeval_seeds=args.reeval_seeds,
           sigma_init=args.sigma_init,
           sigma_min=args.sigma_min,
           seed_pool_size=args.seed_pool_size,
           random_seed_base=1000)

if __name__ == "__main__":
    main()
