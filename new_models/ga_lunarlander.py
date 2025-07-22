# ga_lunarlander_adaptive_mutation.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
from yolo_state import YoloStateEstimator

import time, os
from datetime import datetime
import torch, torch.nn as nn
import numpy as np
import gymnasium as gym
import cv2
from tqdm import trange, tqdm
import matplotlib.pyplot as plt

# === 可调 GA 参数 ===
POP = 4
KEEP_RATIO = 0.10
N_GEN = 1
EPISODES = 10
HIGH_MUT_RATE = 0.1
LOW_MUT_RATE = 0.02
MUTATION_TYPE = 'adaptive'
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

def evaluate_ind(vec):
    total_reward = 0.0
    for _ in range(EPISODES):
        obs, _ = env.reset()
        done = False
        while not done:
            frame = env.render()
            # Gym output is RGB → convert to BGR for YOLO consistency
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            yolo_obs = YOLO_ESTIMATOR.update(frame)
            if yolo_obs is None:
                total_reward += -100
                break

            state = torch.tensor(yolo_obs, dtype=torch.float32)
            action = int(torch.argmax(model(state)).item())
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            total_reward += r

    land_ok = YOLO_ESTIMATOR.frames_lander_ok
    terr_ok = YOLO_ESTIMATOR.frames_terrain_ok
    tot = YOLO_ESTIMATOR.frames_total
    if tot > 0:
        p1 = 100 * land_ok / tot
        p2 = 100 * terr_ok / tot
        print(f"[INFO] Lander识别成功率：{p1:.1f}%，Terrain识别成功率：{p2:.1f}%")

    YOLO_ESTIMATOR.frames_total = YOLO_ESTIMATOR.frames_lander_ok = YOLO_ESTIMATOR.frames_terrain_ok = 0
    return total_reward / EPISODES

def shutdown_after(delay):
    os.system(f"shutdown -s -t {delay}")

def main():
    global env, model, YOLO_ESTIMATOR

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    model = NNPolicy()

    YOLO_ESTIMATOR = YoloStateEstimator(
        lander_model_path="./best_lander_only.pt",
        terrain_model_path="./terrain.pt",
        conf=0.67
    )

    dim = sum(p.numel() for p in model.parameters())
    pop = [np.random.randn(dim) for _ in range(POP)]
    rewards = {'best':[], 'mean':[], 'worst':[]}

    start = time.time()
    t0 = datetime.now().strftime("%Y%m%d_%H%M%S")

    for gen in trange(N_GEN, desc="GA Training"):
        fitness = []
        for vec in pop:
            set_weights_vector(model, vec)
            fitness.append(evaluate_ind(vec))
        f = np.array(fitness)
        avg_f, best, worst = f.mean(), f.max(), f.min()
        rewards['best'].append(best); rewards['mean'].append(avg_f); rewards['worst'].append(worst)
        tqdm.write(f"Gen {gen+1}: best={best:.1f}, mean={avg_f:.1f}, worst={worst:.1f}")

        surv = [pop[i] for i in np.argsort(f)[-max(2, int(POP * KEEP_RATIO)):]]
        new_pop = surv.copy()

        while len(new_pop) < POP:
            i1, i2 = np.random.choice(len(surv), 2, replace=False)
            c1, c2 = uniform_crossover(surv[i1], surv[i2])
            rate1 = HIGH_MUT_RATE if f[i1] < avg_f else LOW_MUT_RATE
            rate2 = HIGH_MUT_RATE if f[i2] < avg_f else LOW_MUT_RATE
            new_pop.append(mutate(c1, rate1))
            if len(new_pop) < POP:
                new_pop.append(mutate(c2, rate2))
        pop = new_pop

    elapsed = time.time() - start
    print(f"训练完成，总耗时 {elapsed:.1f} 秒。")
    best_idx = int(np.argmax(f))
    best_vec = pop[best_idx]
    set_weights_vector(model, best_vec)
    wfn = f"best_weights_{t0}.pth"
    torch.save(model.state_dict(), wfn)
    print(f"✅ 已保存最佳模型权重：{wfn}")

    figfn = f"training_rewards_{t0}.png"
    plt.figure(figsize=(10,6))
    gens = list(range(1, N_GEN+1))
    plt.plot(gens, rewards['mean'], label='Mean')
    plt.plot(gens, rewards['best'], label='Best')
    plt.fill_between(gens, rewards['worst'], rewards['best'], color='gray', alpha=0.2)
    plt.xlabel('Generation'); plt.ylabel('Reward'); plt.title('GA Training Rewards')
    plt.legend(); plt.grid(True)
    plt.savefig(figfn); plt.show()
    print(f"📈 已保存训练曲线图：{figfn}")

    if elapsed > SHUTDOWN_DELAY:
        print(f"训练超过 {SHUTDOWN_DELAY} 秒，{SHUTDOWN_DELAY} 秒后自动关机；取消命令：shutdown -a")
        shutdown_after(SHUTDOWN_DELAY)
    else:
        print("训练时间不足，已跳过自动关机。")

if __name__ == "__main__":
    main()
