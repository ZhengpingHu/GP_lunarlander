import torch
import numpy as np
import gymnasium as gym
from model import FixedLanderNet

def evaluate_individual(param_vector, structure, episodes=3,
                        window_size=4, state_dim=8, action_dim=4,
                        seed=None, render=False):

    env = gym.make("LunarLander-v3")
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    model = FixedLanderNet(window_size=window_size,
                           state_dim=state_dim,
                           action_dim=action_dim,
                           hidden_dims=structure[1:-1],
                           output_dim=structure[-1])
    model.set_flat_params(param_vector)
    model.eval()

    total_reward = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False

        # 初始化缓冲区（状态与logits）
        state_buffer = [np.zeros(state_dim, dtype=np.float32) for _ in range(window_size)]
        output_buffer = [np.zeros(action_dim, dtype=np.float32) for _ in range(window_size)]

        while not done:
            # 更新状态缓冲区
            state_buffer.pop(0)
            state_buffer.append(obs)

            # 构建输入
            input_vec = np.concatenate(state_buffer + output_buffer)
            input_tensor = torch.tensor(input_vec, dtype=torch.float32)

            # 推理
            with torch.no_grad():
                logits = model(input_tensor)
                action = torch.argmax(logits).item()

            # 更新动作输出缓冲区
            output_buffer.pop(0)
            output_buffer.append(logits.numpy())

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if render:
                env.render()

    env.close()
    return total_reward / episodes


def evaluate_batch(params_list, structure, episodes=3, n_workers=4,
                   window_size=4, state_dim=8, action_dim=4):
    """
    多进程评估多个个体
    """
    from multiprocessing import Pool

    args = [
        (params, structure, episodes, window_size, state_dim, action_dim)
        for params in params_list
    ]
    with Pool(n_workers) as pool:
        fitnesses = pool.starmap(evaluate_individual, args)
    return fitnesses
