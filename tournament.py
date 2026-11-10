import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
import numpy as np

from config import BOX_COUNT, MAX_TURNS_PER_GAME
from game.agent import Agent
from ai.baseline_agent import BaselineAgent
from ai.evaluation import evaluate_agent_batched
from ai.circular_neural_network import _numba_conv_forward
from game.batch_game import _numba_step_kernel
from game.game import Game

# Anchor all paths to this script's directory so it works on any computer/IDE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def warm_up_numba():
    """Wakes up the Numba JIT compiler before evaluation begins."""
    dummy_input = np.zeros((1, 1, BOX_COUNT), dtype=np.float64)
    dummy_weight = np.zeros((8, 1, 3), dtype=np.float64)
    dummy_bias = np.zeros(8, dtype=np.float64)
    dummy_indices = np.zeros((BOX_COUNT, 3), dtype=np.int64)
    _numba_conv_forward(dummy_input, dummy_weight, dummy_bias, dummy_indices)

    dummy_pos = np.zeros(10, dtype=np.int64)
    dummy_probs = np.ones((10, BOX_COUNT), dtype=np.float64) / BOX_COUNT
    dummy_turns = np.zeros(10, dtype=np.float64)
    dummy_fin = np.zeros(10, dtype=np.bool_)
    dummy_guesses = np.zeros(10, dtype=np.int64)
    dummy_rng = np.ones(10, dtype=np.int64)
    _numba_step_kernel(dummy_pos, dummy_probs, dummy_turns, dummy_fin, dummy_guesses, dummy_rng, None)


def evaluate_shared_worker(args):
    """Worker function to evaluate a single champion genome using a shared test set seed."""
    shm_name, shape, dtype, games, seed_offset, agent_idx = args

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    genomes = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    genome = genomes[agent_idx].copy()
    existing_shm.close()

    agent = Agent(network_type="circular")
    agent.network.set_genome(genome)

    return evaluate_agent_batched(agent, games=games, seed_offset=seed_offset)


def evaluate_baseline_parallel(games_count, seed_offset):
    """Evaluates the baseline greedy agent across the shared test games."""
    baseline = BaselineAgent()
    turns_list = []
    for g in range(games_count):
        baseline.reset(seed=seed_offset + g)
        game = Game(seed=seed_offset + g)
        turns = 0
        while not game.finished:
            if turns >= MAX_TURNS_PER_GAME:
                turns = MAX_TURNS_PER_GAME
                break
            turns += 1
            guess = baseline.choose_guess()
            found = game.guess(guess)
            if found:
                break
            baseline.update(guess)
        turns_list.append(turns)
    return np.mean(turns_list)


def main():
    print("========== MULTI-TRIAL ROBUST CHAMPION TOURNAMENT ==========")
    warm_up_numba()

    games_per_trial = 10000  # 10k games per trial
    num_trials = 5           # Number of independent random test suites to average over

    print(f"Tournament Configuration: {num_trials} trials | {games_per_trial:,} games/trial ({num_trials * games_per_trial:,} total games per agent)")

    # 1. Gather all champion files across all batch folders first
    max_runs_per_batch = 20
    batch_idx = 1
    champion_metadata = []

    while True:
        batch_dir = os.path.join(BASE_DIR, f"batch_{batch_idx}")
        if not os.path.exists(batch_dir):
            break

        for run_idx in range(1, max_runs_per_batch + 1):
            filename = os.path.join(batch_dir, f"run{run_idx}.npz")
            if os.path.exists(filename):
                champion_metadata.append((f"Batch {batch_idx} | Run {run_idx}", filename))

        batch_idx += 1

    if not champion_metadata:
        print("\n[Notice] No champion files found in any batch folders yet! Complete some training runs first.")
        return

    print(f"Discovered {len(champion_metadata)} champions across batch folders.")

    # Prepare shared memory for champions once
    genomes_list = []
    names_list = []
    for name, filepath in champion_metadata:
        data = np.load(filepath)
        genomes_list.append(data["genome"])
        names_list.append(name)

    genomes_arr = np.array(genomes_list, dtype=np.float64)
    population_size = len(genomes_arr)

    shm = shared_memory.SharedMemory(create=True, size=genomes_arr.nbytes)
    shared_genomes = np.ndarray(genomes_arr.shape, dtype=genomes_arr.dtype, buffer=shm.buf)
    shared_genomes[:] = genomes_arr[:]

    # Accumulators for multi-trial scoring
    accumulated_scores = {name: 0.0 for name in names_list}
    accumulated_baseline = 0.0

    workers = os.cpu_count() or 4
    tournament_start = time.perf_counter()

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for trial in range(1, num_trials + 1):
                # Generate a fresh random seed offset for this trial
                seed_offset = int(np.random.randint(1, 10000000))
                print(f"\n--- Running Trial {trial}/{num_trials} (Seed Offset: {seed_offset}) ---")

                # Evaluate Baseline for this trial
                baseline_score = evaluate_baseline_parallel(games_per_trial, seed_offset)
                accumulated_baseline += baseline_score
                print(f"  -> Baseline Trial Score: {baseline_score:.4f}")

                # Evaluate all champions for this trial in parallel
                tasks = [
                    (shm.name, genomes_arr.shape, genomes_arr.dtype, games_per_trial, seed_offset, i)
                    for i in range(population_size)
                ]

                trial_scores = list(
                    executor.map(evaluate_shared_worker, tasks, chunksize=max(1, population_size // (workers * 4)))
                )

                for name, score in zip(names_list, trial_scores):
                    accumulated_scores[name] += score

    finally:
        shm.close()
        shm.unlink()

    total_elapsed = time.perf_counter() - tournament_start
    print(f"\nCompleted {num_trials} trials across {population_size + 1} agents in {total_elapsed:.2f}s.")

    # 2. Compute true averages across all trials
    results = [("Baseline Greedy", accumulated_baseline / num_trials)]
    for name in names_list:
        avg_score = accumulated_scores[name] / num_trials
        results.append((name, avg_score))

    # 3. Sort and display final robust leaderboard
    results.sort(key=lambda x: x[1])

    print(f"\n================ ROBUST MULTI-TRIAL LEADERBOARD ({num_trials} Trials) ================")
    print(f"{'Rank':<6} | {'Agent Name':<20} | {'True Average Guesses'}")
    print("-" * 57)
    for rank, (name, score) in enumerate(results, 1):
        print(f"{rank:<6} | {name:<20} | {score:.4f}")
    print("=======================================================================")


if __name__ == "__main__":
    main()