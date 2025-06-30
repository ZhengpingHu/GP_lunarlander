import torch
import random
import numpy as np
from copy import deepcopy
from model import FixedLanderNet

def init_population(pop_size, structure):
    """初始化种群，每个个体是扁平参数 tensor"""
    dummy_net = FixedLanderNet(structure)
    param_count = dummy_net.get_param_count()
    return [torch.randn(param_count) * 0.1 for _ in range(pop_size)]

def tournament_selection(population, fitnesses, k=3):
    """锦标赛选择"""
    selected = []
    for _ in range(len(population)):
        candidates = random.sample(list(zip(population, fitnesses)), k)
        winner = max(candidates, key=lambda x: x[1])
        selected.append(winner[0].clone())
    return selected

def uniform_crossover(p1, p2):
    """均匀交叉"""
    mask = torch.rand_like(p1) < 0.5
    child = torch.where(mask, p1, p2)
    return child

def mutate(p, mutation_rate=0.05, mutation_scale=0.1):
    """突变操作：高斯扰动"""
    mask = (torch.rand_like(p) < mutation_rate)
    noise = torch.randn_like(p) * mutation_scale
    p[mask] += noise[mask]
    return p

def next_generation(elites, offspring_count, mutation_rate, mutation_scale):
    """生成下一代个体"""
    next_pop = elites.copy()
    while len(next_pop) < offspring_count:
        p1, p2 = random.choices(elites, k=2)
        child = uniform_crossover(p1, p2)
        child = mutate(child, mutation_rate, mutation_scale)
        next_pop.append(child)
    return next_pop

class EarlyStopper:
    def __init__(self, patience=10, min_delta=1.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float('inf')
        self.counter = 0

    def step(self, score):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
