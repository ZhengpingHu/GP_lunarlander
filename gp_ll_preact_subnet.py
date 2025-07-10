# ga_lunarlander_with_action_subnet.py

import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*", category=DeprecationWarning)

import time, os, multiprocessing as mp
from datetime import datetime
import torch, torch.nn as nn
import numpy as np
import gymnasium as gym
from tqdm import trange, tqdm
import matplotlib.pyplot as plt

# === GA 参数（集中管理）===
POP = 500
KEEP_RATIO = 0.10
N_GEN = 600
EPISODES = 10
HIGH_MUT_RATE = 0.1
LOW_MUT_RATE = 0.02
MUTATION_TYPE = 'adaptive'
SHUTDOWN_DELAY = 60

# === NNPolicy：含专门子网处理 prev_action ===
class NNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # 子网 A: 前4维状态
        self.netA = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        # 子网 B: 后4维状态
        self.netB = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        # 子网 C: 处理 prior action one-hot
        self.netC = nn.Sequential(nn.Linear(4, 16), nn.ReLU())
        # 融合层：拼接3个子网输出 + 合并逻辑
        self.fusion = nn.Sequential(nn.Linear(32*2 + 16, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, state, action_vec):
        featA = self.netA(state[..., :4])
        featB = self.netB(state[..., 4:])
        featC = self.netC(action_vec)
        return self.fusion(torch.cat([featA, featB, featC], dim=-1))

# === 遗传操作及向量化 ===
def get_weights_vector(m):
    return torch.cat([p.data.flatten() for p in m.parameters()]).numpy()

def set_weights_vector(m, vec):
    ptr = 0
    for p in m.parameters():
        num = p.numel()
        p.data.copy_(torch.from_numpy(vec[ptr:ptr+num]).view(p.shape))
        ptr += num

def uniform_crossover(p1, p2):
    mask = np.random.rand(len(p1)) < 0.5
    return np.where(mask, p1, p2), np.where(~mask, p1, p2)

def mutate(vec, rate):
    return vec + np.random.randn(len(vec)) * rate

# === 并行评估函数，带 action 子网处理 ===
def evaluate_ind(args):
    vec, episodes = args
    env = gym.make("LunarLander-v3")
    model = NNPolicy()
    set_weights_vector(model, vec)
    total_r = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        prev_action = np.zeros(4, dtype=np.float32)
        done = False
        while not done:
            state = torch.tensor(obs, dtype=torch.float32)
            action_vec = torch.tensor(prev_action, dtype=torch.float32)
            logits = model(state.unsqueeze(0), action_vec.unsqueeze(0)).squeeze(0)
            action = int(torch.argmax(logits).item())
            prev_action = np.zeros(4, dtype=np.float32)
            prev_action[action] = 1.0
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            total_r += r
    env.close()
    return total_r / episodes

def shutdown_after(delay):
    os.system(f"shutdown -s -t {delay}")

# === 主训练流程（保留原来完整逻辑）===
def main():
    workers = max(mp.cpu_count() - 2, 1)
    pool = mp.Pool(workers)
    print(f"使用 {workers} 个并行评估进程。")

    model = NNPolicy()
    dim = sum(p.numel() for p in model.parameters())
    pop = [np.random.randn(dim) for _ in range(POP)]

    start = time.time()
    t0 = datetime.now().strftime("%Y%m%d_%H%M%S")
    rewards = {'best': [], 'mean': [], 'worst': []}

    for gen in trange(N_GEN, desc="GA Training"):
        fitness = pool.map(evaluate_ind, [(vec, EPISODES) for vec in pop])
        f = np.array(fitness)
        avg_f = f.mean()
        best, mean, worst = f.max(), f.mean(), f.min()
        rewards['best'].append(best)
        rewards['mean'].append(mean)
        rewards['worst'].append(worst)
        tqdm.write(f"Gen {gen+1}: best={best:.1f}, mean={mean:.1f}, worst={worst:.1f}")

        keep_n = int(POP * KEEP_RATIO)
        surv = [pop[i] for i in np.argsort(f)[-keep_n:]]
        new_pop = surv.copy()
        while len(new_pop) < POP:
            i1, i2 = np.random.choice(len(surv), 2, replace=False)
            c1, c2 = uniform_crossover(surv[i1], surv[i2])
            if MUTATION_TYPE == 'adaptive':
                rate1 = HIGH_MUT_RATE if f[i1] < avg_f else LOW_MUT_RATE
                rate2 = HIGH_MUT_RATE if f[i2] < avg_f else LOW_MUT_RATE
            else:
                rate1 = rate2 = LOW_MUT_RATE
            new_pop.append(mutate(c1, rate1))
            if len(new_pop) < POP:
                new_pop.append(mutate(c2, rate2))
        pop = new_pop

    pool.close()
    elapsed = time.time() - start
    print(f"训练完成，总耗时 {elapsed:.1f} 秒。")

    best_idx = int(np.argmax(f))
    best_vec = pop[best_idx]
    set_weights_vector(model, best_vec)
    wfn = f"best_weights_{t0}.pth"
    torch.save(model.state_dict(), wfn)
    print(f"✅ 保存最佳权重文件：{wfn}")

    figfn = f"training_rewards_{t0}.png"
    plt.figure(figsize=(10, 6))
    gens = list(range(1, N_GEN+1))
    plt.plot(gens, rewards['mean'], label='Mean')
    plt.plot(gens, rewards['best'], label='Best')
    plt.fill_between(gens, rewards['worst'], rewards['best'], color='gray', alpha=0.2)
    plt.xlabel('Generation')
    plt.ylabel('Reward')
    plt.title('GA Training Rewards')
    plt.legend()
    plt.grid(True)
    plt.savefig(figfn)
    plt.show()
    print(f"📈 保存训练曲线图：{figfn}")

    if elapsed > SHUTDOWN_DELAY:
        print(f"训练时间 > {SHUTDOWN_DELAY} 秒，{SHUTDOWN_DELAY} 秒后关机；取消命令：shutdown -a")
        shutdown_after(SHUTDOWN_DELAY)
    else:
        print("训练时间未达阈值，跳过自动关机。")

if __name__ == "__main__":
    main()