# ga_lunarlander_with_prev_action_both_subnets.py

import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*", category=DeprecationWarning)

import time, os, multiprocessing as mp
from datetime import datetime
import torch, torch.nn as nn
import numpy as np
import gymnasium as gym
from tqdm import trange, tqdm
import matplotlib.pyplot as plt

# === GA 参数（集中管理） ===
POP = 500
KEEP_RATIO = 0.10
N_GEN = 600
EPISODES = 10
HIGH_MUT_RATE = 0.1
LOW_MUT_RATE = 0.02
MUTATION_TYPE = 'adaptive'
SHUTDOWN_DELAY = 60

# === NNPolicy 网络：两个子网均接收状态+前一步动作信息 ===
class NNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # 子网 A: 接收 state[0:4] + full prev_action（4）→ 输入 8
        self.netA = nn.Sequential(nn.Linear(8, 32), nn.ReLU())
        # 子网 B: 接收 state[4:8] + full prev_action（4）→ 输入 8
        self.netB = nn.Sequential(nn.Linear(8, 32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):
        state = x[..., :8]
        action_vec = x[..., 8:]
        a = torch.cat([state[..., :4], action_vec], dim=-1)   # 子网 A 输入
        b = torch.cat([state[..., 4:], action_vec], dim=-1)   # 子网 B 输入
        featA, featB = self.netA(a), self.netB(b)
        return self.fusion(torch.cat([featA, featB], dim=-1))

# === 遗传算法中常用函数 ===
def get_weights_vector(model):
    return torch.cat([p.data.flatten() for p in model.parameters()]).numpy()

def set_weights_vector(model, vec):
    ptr = 0
    for p in model.parameters():
        num = p.numel()
        p.data.copy_(torch.from_numpy(vec[ptr:ptr+num]).view(p.shape))
        ptr += num

def uniform_crossover(p1, p2):
    mask = np.random.rand(len(p1)) < 0.5
    return np.where(mask, p1, p2), np.where(~mask, p1, p2)

def mutate(vec, rate):
    return vec + np.random.randn(len(vec)) * rate

# === 并行评估函数：加入 prev_action 递归输入 ===
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
            inp = np.concatenate([obs, prev_action])
            logits = model(torch.tensor(inp, dtype=torch.float32))
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

# === 主流程 ===
def main():
    workers = max(mp.cpu_count() - 2, 1)
    pool = mp.Pool(workers)
    print(f"Using {workers} parallel evaluators.")

    model = NNPolicy()
    dim = sum(p.numel() for p in model.parameters())
    pop = [np.random.randn(dim) for _ in range(POP)]

    start = time.time()
    t0 = datetime.now().strftime("%Y%m%d_%H%M%S")
    rewards = {'best': [], 'mean': [], 'worst': []}

    for gen in trange(N_GEN, desc="GA Training"):
        fitness = pool.map(evaluate_ind, [(vec, EPISODES) for vec in pop])
        f = np.array(fitness)
        avg_f, best, mean, worst = f.mean(), f.max(), f.mean(), f.min()
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
    print(f"Training finished in {elapsed:.1f}s.")

    f_last = f
    best_idx = int(np.argmax(f_last))
    best_vec = pop[best_idx]
    set_weights_vector(model, best_vec)
    wfn = f"best_weights_{t0}.pth"
    torch.save(model.state_dict(), wfn)
    print(f"✅ Saved weights: {wfn}")

    figfn = f"training_rewards_{t0}.png"
    plt.figure(figsize=(10, 6))
    gens = list(range(1, N_GEN+1))
    plt.plot(gens, rewards['mean'], label='Mean')
    plt.plot(gens, rewards['best'], label='Best')
    plt.fill_between(gens, rewards['worst'], rewards['best'], color='gray', alpha=0.2)
    plt.xlabel('Generation'); plt.ylabel('Reward'); plt.title('GA Training Rewards')
    plt.legend(); plt.grid(True)
    plt.savefig(figfn); plt.show()
    print(f"📈 Plot saved: {figfn}")

    if elapsed > SHUTDOWN_DELAY:
        print(f"Training > {SHUTDOWN_DELAY}s, shutting down in {SHUTDOWN_DELAY}s. Cancel with: shutdown -a")
        shutdown_after(SHUTDOWN_DELAY)
    else:
        print("Training time < threshold, skipping shutdown.")

if __name__ == "__main__":
    main()
