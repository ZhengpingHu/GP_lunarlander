import gym
import torch
from model import FixedLanderNet

def evaluate_individual(param_vector, structure, episodes=3, seed=None, render=False):
    """
    评估一个参数向量在环境中的表现（平均 reward）
    """
    env = gym.make("LunarLander-v3")
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    model = FixedLanderNet(structure)
    model.set_flat_params(param_vector)
    model.eval()

    total_reward = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                logits = model(obs_tensor)
                action = torch.argmax(logits).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if render:
                env.render()
    env.close()
    return total_reward / episodes

def evaluate_batch(params_list, structure, episodes=3, n_workers=4):
    """
    多进程评估一批个体的平均 reward
    """
    from multiprocessing import Pool
    args = [(params, structure, episodes) for params in params_list]
    with Pool(n_workers) as pool:
        fitnesses = pool.starmap(evaluate_individual, args)
    return fitnesses
