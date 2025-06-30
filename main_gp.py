import argparse
import os
import logging
from tqdm import tqdm

from configs import *
from gp_core import *
from eval_worker import evaluate_batch, evaluate_individual
from utils import (
    set_seed, save_model, load_model,
    save_checkpoint, load_checkpoint, setup_logger
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--structure', type=int, nargs='+',
                        default=DEFAULT_STRUCTURE, help="Net structure like 8 32 16 4")
    parser.add_argument('--pop-size', type=int, default=DEFAULT_POP_SIZE)
    parser.add_argument('--gen', type=int, default=DEFAULT_N_GENERATIONS)
    parser.add_argument('--n-workers', type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument('--resume', action='store_true', help="Resume from checkpoint")
    parser.add_argument('--test', action='store_true', help="Evaluate saved model")
    parser.add_argument('--load-path', type=str, default=None, help="Path to .pt model to test")
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def train(args):
    set_seed(args.seed)
    setup_logger()
    os.makedirs("checkpoint", exist_ok=True)

    structure = args.structure
    pop_size = args.pop_size
    n_workers = args.n_workers
    max_gen = args.gen

    # 断点恢复
    if args.resume and os.path.exists(CHECKPOINT_PATH):
        logging.info("Loading checkpoint...")
        population, fitnesses, start_gen, best_reward = load_checkpoint(CHECKPOINT_PATH)
    else:
        population = init_population(pop_size, structure)
        fitnesses = [0.0] * pop_size
        start_gen = 0
        best_reward = -float('inf')

    stopper = EarlyStopper(EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA)

    for gen in range(start_gen, max_gen):
        logging.info(f"Generation {gen} evaluating...")
        fitnesses = evaluate_batch(population, structure,
                                   episodes=DEFAULT_EPISODES_PER_INDIVIDUAL,
                                   n_workers=n_workers)

        # 排序
        ranked = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
        population = [p for p, _ in ranked]
        fitnesses = [f for _, f in ranked]

        logging.info(f"  Best fitness: {fitnesses[0]:.2f}, Mean: {sum(fitnesses)/len(fitnesses):.2f}")

        # 保存模型与状态
        save_model(population[0], structure, BEST_MODEL_PATH_TEMPLATE.format(gen=gen))
        save_checkpoint(CHECKPOINT_PATH, population, fitnesses, gen+1, fitnesses[0])

        if fitnesses[0] > best_reward:
            best_reward = fitnesses[0]
            save_model(population[0], structure, FINAL_MODEL_PATH)

        # Early stopping
        if stopper.step(fitnesses[0]):
            logging.info("Early stopping triggered.")
            break

        # 精英保留 + 演化下一代
        n_elite = max(1, int(pop_size * DEFAULT_ELITE_FRACTION))
        elites = population[:n_elite]
        population = next_generation(elites, pop_size,
                                     DEFAULT_MUTATION_RATE, DEFAULT_MUTATION_SCALE)

    logging.info("Training completed. Final model saved.")

def test(args):
    from eval_worker import evaluate_individual
    assert args.load_path is not None, "Must provide --load-path to test"
    model = load_model(args.load_path, args.structure)
    score = evaluate_individual(model.get_flat_params(), args.structure, episodes=5, render=True)
    print(f"Average reward over 5 episodes: {score:.2f}")

if __name__ == "__main__":
    args = parse_args()
    if args.test:
        test(args)
    else:
        train(args)
