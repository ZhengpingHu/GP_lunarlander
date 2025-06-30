import argparse
import os
import logging
import csv
import matplotlib.pyplot as plt
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
                        default=None, help="Net structure like 120 64 32 4 (auto if omitted)")
    parser.add_argument('--pop-size', type=int, default=DEFAULT_POP_SIZE)
    parser.add_argument('--gen', type=int, default=DEFAULT_N_GENERATIONS)
    parser.add_argument('--n-workers', type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument('--resume', action='store_true', help="Resume from checkpoint")
    parser.add_argument('--test', action='store_true', help="Evaluate saved model")
    parser.add_argument('--load-path', type=str, default=None, help="Path to .pt model to test")
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def get_structure(struct_override):
    if struct_override:
        return struct_override
    input_dim = WINDOW_SIZE * (STATE_DIM + ACTION_DIM)
    return [input_dim, 64, 32, ACTION_DIM]


def plot_reward_curve(log_data, out_path="checkpoint/reward_curve.png"):
    gens = [x[0] for x in log_data]
    bests = [x[1] for x in log_data]
    means = [x[2] for x in log_data]
    worsts = [x[3] for x in log_data]

    plt.figure(figsize=(10, 6))
    plt.plot(gens, bests, label="Best", linewidth=2)
    plt.plot(gens, means, label="Mean", linestyle='--')
    plt.plot(gens, worsts, label="Worst", linestyle=':')
    plt.xlabel("Generation")
    plt.ylabel("Reward")
    plt.title("GP Performance over Generations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()


def train(args):
    set_seed(args.seed)
    setup_logger()
    os.makedirs("checkpoint", exist_ok=True)

    structure = get_structure(args.structure)
    pop_size = args.pop_size
    n_workers = args.n_workers
    max_gen = args.gen

    log_path = "checkpoint/reward_log.csv"
    reward_log = []

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best", "mean", "worst"])

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

        fitnesses = []
        for i in tqdm(range(pop_size), desc=f"Evaluating Gen {gen}"):
            score = evaluate_individual(
                population[i], structure,
                episodes=DEFAULT_EPISODES_PER_INDIVIDUAL,
                window_size=WINDOW_SIZE,
                state_dim=STATE_DIM,
                action_dim=ACTION_DIM
            )
            fitnesses.append(score)

        ranked = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
        population = [p for p, _ in ranked]
        fitnesses = [f for _, f in ranked]

        best = fitnesses[0]
        mean = sum(fitnesses) / len(fitnesses)
        worst = fitnesses[-1]

        logging.info(f"  Best fitness: {best:.2f}, Mean: {mean:.2f}, Worst: {worst:.2f}")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([gen, best, mean, worst])

        reward_log.append((gen, best, mean, worst))

        save_model(population[0], structure, BEST_MODEL_PATH_TEMPLATE.format(gen=gen))
        save_checkpoint(CHECKPOINT_PATH, population, fitnesses, gen + 1, best)

        if best > best_reward:
            best_reward = best
            save_model(population[0], structure, FINAL_MODEL_PATH)

        if stopper.step(best):
            logging.info("Early stopping triggered.")
            break

        n_elite = max(1, int(pop_size * DEFAULT_ELITE_FRACTION))
        elites = population[:n_elite]
        population = next_generation(elites, pop_size,
                                     DEFAULT_MUTATION_RATE, DEFAULT_MUTATION_SCALE)

    logging.info("Training completed. Final model saved.")
    plot_reward_curve(reward_log)


def test(args):
    structure = get_structure(args.structure)
    assert args.load_path is not None, "Must provide --load-path to test"
    model = load_model(args.load_path, structure)
    score = evaluate_individual(
        model.get_flat_params(),
        structure,
        episodes=5,
        render=True,
        window_size=WINDOW_SIZE,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM
    )
    print(f"Average reward over 5 episodes: {score:.2f}")


if __name__ == "__main__":
    args = parse_args()
    if args.test:
        test(args)
    else:
        train(args)
