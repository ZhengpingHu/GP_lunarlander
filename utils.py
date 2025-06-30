import os
import torch
import pickle
import random
import numpy as np
import logging
from configs import WINDOW_SIZE, STATE_DIM, ACTION_DIM
from model import FixedLanderNet

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_model(param_vector, structure, path):
    """将神经网络参数保存为 .pt 文件"""
    model = FixedLanderNet(
        window_size=WINDOW_SIZE,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dims=structure[1:-1],
        output_dim=structure[-1]
    )
    model.set_flat_params(param_vector)
    torch.save(model.state_dict(), path)

def load_model(path, structure):
    """从 .pt 文件加载神经网络权重"""
    model = FixedLanderNet(
        window_size=WINDOW_SIZE,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dims=structure[1:-1],
        output_dim=structure[-1]
    )
    model.load_state_dict(torch.load(path))
    return model

def save_checkpoint(path, population, fitnesses, generation, best_reward):
    """保存 checkpoint（含完整状态）"""
    state = {
        "population": population,
        "fitnesses": fitnesses,
        "generation": generation,
        "best_reward": best_reward
    }
    with open(path, "wb") as f:
        pickle.dump(state, f)

def load_checkpoint(path):
    """加载 checkpoint（恢复训练）"""
    with open(path, "rb") as f:
        state = pickle.load(f)
    return (state["population"], state["fitnesses"],
            state["generation"], state["best_reward"])

def setup_logger():
    """初始化日志格式"""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO
    )
