#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA（4D 输入） + 远程 YOLO RPC 客户端
- 每个 worker 建立一次“长连接”并全程复用
- 断线自动指数回退重连
- 支持 infer / predict / reset
- 打印真实 global-best 和 avg-top10%
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # 训练端不用 CUDA 推理

import time
import struct
import pickle
import random
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import cv2
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.connection import Client


# ----------------- GA 超参（小批量更快观测趋势） -----------------
POP = 120                 # 种群
GEN = 60                  # 代数
EPISODES = 4              # 每个体评估 4 个 seed
FRAMES_PER_EP = 400       # 每个 episode 最多 400 帧
YOLO_FREQ = 8             # 每 8 帧 infer，一般帧内用 predict
KEEP_RATIO = 0.20
MUT_RATE = 0.02

PENALTY_NO_DET = -50.0    # 三次连续检测失败则中止本 ep 并惩罚
MAX_FAIL_INFER = 3

# ----------------- RPC 参数 -----------------
RPC_ADDR = ("127.0.0.1", 6000)
RPC_AUTH = b"yolo-rpc"


# ----------------- 简单 4D 策略网络 -----------------
class NNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.net(x)


def get_weights_vector(m: nn.Module) -> np.ndarray:
    return torch.cat([p.data.flatten() for p in m.parameters()]).cpu().numpy()


def set_weights_vector(m: nn.Module, vec: np.ndarray) -> None:
    ptr = 0
    for p in m.parameters():
        n = p.numel()
        p.data.copy_(torch.from_numpy(vec[ptr:ptr+n]).view_as(p))
        ptr += n


# ----------------- RPC 客户端（长连接 + 重试 + 断线自愈） -----------------
class RemoteEstimator:
    def __init__(self, address: Tuple[str, int], authkey: bytes):
        self.address = address
        self.authkey = authkey
        self.conn: Optional[Client] = None

    def _ensure(self) -> None:
        if self.conn is not None:
            return
        last_exc = None
        for i in range(8):  # 指数回退
            try:
                self.conn = Client(self.address, authkey=self.authkey)
                return
            except Exception as e:
                last_exc = e
                time.sleep(0.05 * (2 ** i))
        raise last_exc

    def close(self) -> None:
        try:
            if self.conn is not None:
                self.conn.close()
        except Exception:
            pass
        self.conn = None

    def _send(self, d: dict) -> None:
        try:
            self.conn.send(d)
        except Exception:
            # 断线重连一次
            self.close()
            self._ensure()
            self.conn.send(d)

    def _recv(self) -> Optional[dict]:
        try:
            obj = self.conn.recv()
            if isinstance(obj, dict):
                return obj
            return None
        except Exception:
            # 交给上层决定是否重发
            self.close()
            return None

    def ping(self) -> bool:
        self._ensure()
        self._send({"cmd": "ping"})
        r = self._recv()
        return bool(r and r.get("ok", False))

    def reset(self) -> None:
        self._ensure()
        self._send({"cmd": "reset"})
        _ = self._recv()

    def infer(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        self._ensure()
        self._send({"cmd": "infer", "frame": frame_bgr})
        r = self._recv()
        if r and r.get("ok", False):
            return np.array(r["z"], dtype=np.float32)
        return None

    def predict(self) -> Optional[np.ndarray]:
        self._ensure()
        self._send({"cmd": "predict"})
        r = self._recv()
        if r and r.get("ok", False):
            return np.array(r["z"], dtype=np.float32)
        return None


# 让每个 worker 进程持有一个“长连接”
_RPC: Optional[RemoteEstimator] = None
def _get_rpc() -> RemoteEstimator:
    global _RPC
    if _RPC is None:
        _RPC = RemoteEstimator(RPC_ADDR, RPC_AUTH)
    return _RPC


# ----------------- 单个个体评估 -----------------
def _evaluate_one_episode(model: NNPolicy, seed: int) -> float:
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)

    rpc = _get_rpc()
    rpc.reset()
    fail = 0

    total_reward = 0.0

    for t in range(FRAMES_PER_EP):
        # 渲染帧（RGB）
        frame = env.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if t % YOLO_FREQ == 0:
            z = rpc.infer(frame_bgr)
        else:
            z = rpc.predict()

        if z is None:
            fail += 1
            if fail >= MAX_FAIL_INFER:
                total_reward += PENALTY_NO_DET
                break
            # 没有检测到就维持不作为（或随机小动作）
            action = env.action_space.sample()
        else:
            fail = 0
            # z: [cosθ, sinθ, x_norm, y_norm]
            with torch.no_grad():
                a = torch.argmax(model(torch.tensor(z, dtype=torch.float32))).item()
            action = int(a)

        _, r, term, trunc, _ = env.step(action)
        total_reward += float(r)
        if term or trunc:
            break

    env.close()
    return total_reward


@dataclass
class Job:
    idx: int
    weights: np.ndarray
    seed: int


def evaluate_individual(job: Job) -> Tuple[float, float]:
    # 每个 worker 只设置一次线程数即可
    torch.set_num_threads(1)
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    model = NNPolicy()
    set_weights_vector(model, job.weights)
    model.eval()

    seeds = [job.seed + i * 997 for i in range(EPISODES)]
    rewards = []
    for sd in seeds:
        rew = _evaluate_one_episode(model, sd)
        rewards.append(rew)
    rewards = np.array(rewards, dtype=np.float32)
    return float(rewards.max()), float(rewards.mean())


# ----------------- GA 主循环 -----------------
def uniform_crossover(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.random.rand(len(p1)) < 0.5
    return np.where(mask, p1, p2), np.where(~mask, p1, p2)


def mutate(vec: np.ndarray, rate: float) -> np.ndarray:
    return vec + np.random.randn(len(vec)) * rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 初始种群
    dim = sum(p.numel() for p in NNPolicy().parameters())
    pop = [np.random.randn(dim).astype(np.float32) for _ in range(POP)]

    print(f"🚀 [INFO] Starting GA training with {args.processes} parallel processes.")

    for g in range(1, GEN + 1):
        # 构造任务
        jobs = [Job(i, pop[i], seed=args.seed + 10000 * g + i) for i in range(POP)]

        results: List[Tuple[float, float]] = []
        t0 = time.time()
        with Pool(processes=args.processes) as pool:
            for b, a in tqdm(
                pool.imap_unordered(evaluate_individual, jobs),
                total=len(jobs),
                desc=f"Generation {g}/{GEN} - Evaluating",
            ):
                results.append((b, a))

        bests = np.array([r[0] for r in results], dtype=np.float32)
        avgs  = np.array([r[1] for r in results], dtype=np.float32)

        # —— 真·全局 best/avg top10% 打印 —— #
        global_best = float(bests.max())
        topk = max(1, POP // 10)
        global_avg_topk = float(np.mean(avgs[np.argsort(avgs)[-topk:]]))
        print(f"[GEN {g:03d}] best={global_best:+.2f}  avg-top10%={global_avg_topk:+.2f}  time={time.time()-t0:.1f}s")

        # 选择（双目标，稳定些）
        rank_b = np.argsort(bests)   # best 越大越好
        rank_a = np.argsort(avgs)    # avg 越大越好
        rank = (rank_b + rank_a).argsort()
        n_surv = max(2, int(POP * KEEP_RATIO))
        survivors_idx = rank[-n_surv:]
        survivors = [pop[i] for i in survivors_idx]

        # 产生新种群
        new_pop = survivors.copy()
        while len(new_pop) < POP:
            i, j = np.random.choice(n_surv, 2, replace=False)
            c1, c2 = uniform_crossover(survivors[i], survivors[j])
            new_pop.append(mutate(c1, MUT_RATE))
            if len(new_pop) < POP:
                new_pop.append(mutate(c2, MUT_RATE))
        pop = new_pop


if __name__ == "__main__":
    # Windows 上强制 spawn 更稳
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    main()
