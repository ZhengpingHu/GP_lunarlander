#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
import math
import argparse
import random
import traceback
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from tqdm import tqdm
from multiprocessing.connection import Client

AUTHKEY = b"yolo-rpc"  # ← 和服务端一致

# 让 tqdm 输出更干净
print = lambda *a, **k: (__import__("builtins").print(*a, **{**k, "flush": True}))

# ----------------------------
# 小型 2层 MLP：输入 4D -> 4 动作
# ----------------------------
class NNPolicy(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(),
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
            p.copy_(torch.from_numpy(vec[offset:offset+n]).view_as(p))
            offset += n

def uniform_crossover(p1: np.ndarray, p2: np.ndarray):
    mask = np.random.rand(len(p1)) < 0.5
    return np.where(mask, p1, p2), np.where(~mask, p1, p2)

def mutate(vec: np.ndarray, sigma: float) -> np.ndarray:
    return vec + np.random.randn(vec.size) * sigma if sigma > 0 else vec.copy()

# ----------------------------
# 远程 YOLO 4D 状态估计 RPC 客户端
# ----------------------------
@dataclass
class RemoteEstimator:
    address: Tuple[str, int]
    authkey: bytes = AUTHKEY
    conn: Optional[Client] = None

    def _ensure(self):
        if self.conn is None:
            self.conn = Client(self.address, authkey=self.authkey)

    def ping(self) -> bool:
        self._ensure()
        self.conn.send(("ping", None))
        resp = self.conn.recv()
        return isinstance(resp, tuple) and len(resp) == 2 and resp[0] == "pong"

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

# ----------------------------
# 单回合评估（用环境 seed 固定随机性）
# ----------------------------
def _evaluate_one_episode(model: NNPolicy, rpc: RemoteEstimator, seed: int, max_steps: int = 1000) -> float:
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    try:
        obs, _ = env.reset(seed=seed)
        rpc.reset()
        done = False
        total = 0.0

        # 初始帧
        frame = env.render()
        state = rpc.update_full(frame[..., ::-1])  # RGB->BGR
        if state is None:
            return -1000.0  # 感知完全失败，强负分

        steps = 0
        while not done and steps < max_steps:
            s = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                act = int(torch.argmax(model(s)).item())

            obs, r, term, trunc, _ = env.step(act)
            total += r
            done = term or trunc

            # 轻量 frame-skip：每 N 步才全推理一次，其余 predict_only
            if steps % 8 == 0:
                frame = env.render()
                state = rpc.update_full(frame[..., ::-1])
                if state is None:
                    return total - 300.0  # 中途感知挂了，重罚
            else:
                state = rpc.predict_only()
                if state is None:
                    return total - 200.0

            steps += 1

        return total
    finally:
        env.close()

# ----------------------------
# 多进程评估：粗评 + Top-K 重评（median）
# ----------------------------
def _coarse_eval_job(args):
    vec, seed, rpc_addr = args
    model = NNPolicy()
    set_weights_vector(model, vec)
    rpc = RemoteEstimator(rpc_addr)
    try:
        return _evaluate_one_episode(model, rpc, seed)
    except Exception:
        return -1000.0

def _reeval_job(args):
    vec, seed, rpc_addr = args
    model = NNPolicy()
    set_weights_vector(model, vec)
    rpc = RemoteEstimator(rpc_addr)
    try:
        return _evaluate_one_episode(model, rpc, seed)
    except Exception:
        return -1000.0

# ----------------------------
# GA 主过程（含：CRN、Top-K 重评、退火）
# ----------------------------
def run_ga(
    population_size: int = 120,
    generations: int = 60,
    keep_ratio: float = 0.2,
    processes: int = max(1, (os.cpu_count() or 4) - 2),
    rpc_host: str = "127.0.0.1",
    rpc_port: int = 6000,
    coarse_seeds: int = 1,
    reeval_topk: float = 0.1,
    reeval_seeds: int = 4,
):
    # 预热/握手
    rpc = RemoteEstimator((rpc_host, rpc_port))
    try:
        if not rpc.ping():
            raise RuntimeError("ping returned unexpected response")
    except Exception as e:
        print(f"[FATAL] RPC ping 失败：{e}\n"
              f"请确认服务端在 {rpc_host}:{rpc_port} 运行，且 AUTHKEY 匹配。")
        return

    # 初始化种群
    base = NNPolicy()
    base_vec = get_weights_vector(base)
    dim = base_vec.size
    pop = [mutate(base_vec, sigma=0.1) for _ in range(population_size)]
    sigma = 0.15

    keep_n = max(1, int(population_size * keep_ratio))
    topk_n = max(1, int(population_size * reeval_topk))

    pool = mp.Pool(processes=processes)

    for gen in range(1, generations + 1):
        # 固定粗评的 seeds（CRN）
        seeds = [random.randint(0, 10**6) for _ in range(coarse_seeds)]

        # 粗评
        jobs = []
        for vec in pop:
            for sd in seeds:
                jobs.append((vec, sd, (rpc_host, rpc_port)))

        rets = list(tqdm(pool.imap_unordered(_coarse_eval_job, jobs),
                         total=len(jobs), desc=f"Generation {gen}/{generations} - Coarse"))
        # 聚合
        scores = []
        idx = 0
        for i in range(population_size):
            rs = rets[idx: idx + coarse_seeds]
            idx += coarse_seeds
            scores.append(float(np.mean(rs)))

        # 选出前 topk 做重评估（多随机种子，取中位数）
        order = np.argsort(scores)[::-1]  # 大到小
        topk_idx = order[:topk_n]
        reeval_jobs = []
        for i in topk_idx:
            for _ in range(reeval_seeds):
                reeval_jobs.append((pop[i], random.randint(0, 10**6), (rpc_host, rpc_port)))

        if reeval_jobs:
            rets2 = list(pool.imap_unordered(_reeval_job, reeval_jobs))
            # 写回中位数
            p = 0
            for j, i in enumerate(topk_idx):
                block = rets2[p: p + reeval_seeds]
                p += reeval_seeds
                scores[i] = float(np.median(block))

        # 统计
        best_score = float(np.max(scores))
        avg_top10 = float(np.mean(np.sort(scores)[-max(1, population_size // 10):]))
        print(f"[GEN {gen:03d}] best={best_score:+.2f}  avg-top10%={avg_top10:+.2f}  "
              f"(coarse-best={float(np.max(scores)):+.2f} avg-top10%={avg_top10:+.2f})  sigma={sigma:.3f}")

        # 选拔 + 生成下一代
        elite_idx = order[:keep_n]
        elites = [pop[i] for i in elite_idx]

        next_pop: List[np.ndarray] = []
        next_pop.extend(elites)

        while len(next_pop) < population_size:
            # 父本：在前半里随机挑
            i1, i2 = np.random.choice(elite_idx, size=2, replace=True)
            c1, c2 = uniform_crossover(pop[i1], pop[i2])
            c1 = mutate(c1, sigma)
            next_pop.append(c1)
            if len(next_pop) < population_size:
                c2 = mutate(c2, sigma)
                next_pop.append(c2)

        pop = next_pop

        # 退火
        sigma = max(0.05, sigma * 0.985)

    pool.close()
    pool.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=120, help="Population size")
    parser.add_argument("--generations", type=int, default=60, help="Number of generations")
    parser.add_argument("--keep-ratio", type=float, default=0.2, help="精英/保留比例")
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 4) - 2), help="并行进程数")
    parser.add_argument("--rpc-host", type=str, default="127.0.0.1", help="RPC server host")
    parser.add_argument("--rpc-port", type=int, default=6000, help="RPC server port")
    parser.add_argument("--coarse-seeds", type=int, default=1, help="每个个体的粗粒度评估次数")
    parser.add_argument("--reeval-topk", type=float, default=0.1, help="每代重评估 top-k 百分比")
    parser.add_argument("--reeval-seeds", type=int, default=4, help="重评估时的种子数量")
    args = parser.parse_args()

    run_ga(
        population_size=args.population,
        generations=args.generations,
        keep_ratio=args.keep_ratio,
        processes=args.processes,
        rpc_host=args.rpc_host,
        rpc_port=args.rpc_port,
        coarse_seeds=args.coarse_seeds,
        reeval_topk=args.reeval_topk,
        reeval_seeds=args.reeval_seeds,
    )


if __name__ == "__main__":
    main()
