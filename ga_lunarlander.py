# ga_lunarlander_adaptive_mutation.py
import warnings

# 忽略所有 pkg_resources 相关的 DeprecationWarning
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*", category=DeprecationWarning)



import time, os, multiprocessing as mp
from datetime import datetime
import torch, torch.nn as nn
import numpy as np
import gymnasium as gym
from tqdm import trange, tqdm
import matplotlib.pyplot as plt

# === 可调整 GA 参数（集中管理） ===
POP = 500
KEEP_RATIO = 0.10
N_GEN = 600
EPISODES = 10
HIGH_MUT_RATE = 0.1     # f_i < avg → 高突变探索
LOW_MUT_RATE = 0.02     # f_i ≥ avg → 低突变保优
MUTATION_TYPE = 'adaptive'  # 'fixed' 或 'adaptive'
SHUTDOWN_DELAY = 60

# === NNPolicy 网络结构 ===
class NNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.netA = nn.Sequential(nn.Linear(2,32), nn.ReLU())
        self.netB = nn.Sequential(nn.Linear(6,32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(64,64), nn.ReLU(), nn.Linear(64,4))
    def forward(self, x):
        a, b = x[..., :2], x[..., 2:]
        return self.fusion(torch.cat([self.netA(a), self.netB(b)], dim=-1))

# === 权重向量化和遗传操作函数 ===
def get_weights_vector(m): return torch.cat([p.data.flatten() for p in m.parameters()]).numpy()
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

# === 单个个体评估函数（并行使用）===
def evaluate_ind(args):
    vec, episodes = args
    env = gym.make("LunarLander-v3")
    model = NNPolicy()
    set_weights_vector(model, vec)
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = int(torch.argmax(model(torch.tensor(obs, dtype=torch.float32))).item())
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            total += r
    env.close()
    return total / episodes

def shutdown_after(delay):
    os.system(f"shutdown -s -t {delay}")

# === 主训练流程 ===
def main():
    workers = max(mp.cpu_count() - 2, 1)
    pool = mp.Pool(workers)
    print(f"使用 {workers} 个并行进程评估个体。")

    model = NNPolicy()
    dim = sum(p.numel() for p in model.parameters())
    pop = [np.random.randn(dim) for _ in range(POP)]

    start = time.time()
    t0 = datetime.now().strftime("%Y%m%d_%H%M%S")
    rewards = {'best':[], 'mean':[], 'worst':[]}

    for gen in trange(N_GEN, desc="GA Training"):
        fitness = pool.map(evaluate_ind, [(vec, EPISODES) for vec in pop])
        f = np.array(fitness)
        avg_f = f.mean()
        best, mean, worst = f.max(), f.mean(), f.min()
        rewards['best'].append(best)
        rewards['mean'].append(mean)
        rewards['worst'].append(worst)
        tqdm.write(f"Gen {gen+1}: best={best:.1f}, mean={mean:.1f}, worst={worst:.1f}")

        surv = [pop[i] for i in np.argsort(f)[-int(POP*KEEP_RATIO):]]
        new_pop = surv.copy()
        for vec_i, f_i in zip(surv, sorted(f)[-len(surv):]):
            pass  # survivors carried over

        while len(new_pop) < POP:
            i1, i2 = np.random.choice(len(surv), 2, replace=False)
            c1, c2 = uniform_crossover(surv[i1], surv[i2])
            if MUTATION_TYPE == 'adaptive':
                rate1 = HIGH_MUT_RATE if fitness[i1] < avg_f else LOW_MUT_RATE
                rate2 = HIGH_MUT_RATE if fitness[i2] < avg_f else LOW_MUT_RATE
            else:
                rate1 = rate2 = LOW_MUT_RATE
            new_pop.append(mutate(c1, rate1))
            if len(new_pop) < POP:
                new_pop.append(mutate(c2, rate2))

        pop = new_pop

    pool.close()
    elapsed = time.time() - start
    print(f"训练完成，总耗时 {elapsed:.1f} 秒。")

    # 保存最终最佳个体
    fitness_last = fitness
    best_idx = int(np.argmax(fitness_last))
    best_vec = pop[best_idx]
    set_weights_vector(model, best_vec)
    wfn = f"best_weights_{t0}.pth"
    torch.save(model.state_dict(), wfn)
    print(f"✅ 已保存最佳模型权重：{wfn}")

    # 绘制训练曲线图
    figfn = f"training_rewards_{t0}.png"
    gens = list(range(1, N_GEN+1))
    plt.figure(figsize=(10,6))
    plt.plot(gens, rewards['mean'], label='Mean')
    plt.plot(gens, rewards['best'], label='Best')
    plt.fill_between(gens, rewards['worst'], rewards['best'], color='gray', alpha=0.2)
    plt.xlabel('Generation'); plt.ylabel('Reward'); plt.title('GA Training Rewards')
    plt.legend(); plt.grid(True)
    plt.savefig(figfn)
    plt.show()
    print(f"📈 已保存训练曲线图：{figfn}")

    # 自动关机判断
    if elapsed > SHUTDOWN_DELAY:
        print(f"训练超过 {SHUTDOWN_DELAY} 秒，{SHUTDOWN_DELAY} 秒后自动关机；取消命令：shutdown -a")
        shutdown_after(SHUTDOWN_DELAY)
    else:
        print("训练时间不足，已跳过自动关机。")

if __name__ == "__main__":
    main()
