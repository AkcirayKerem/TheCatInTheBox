import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
import numpy as np

from config import BOX_COUNT, MAX_TURNS_PER_GAME
from game.agent import Agent
from game.game import Game
from ai.circular_neural_network import _numba_conv_forward
from game.batch_game import _numba_step_kernel
from ai.evaluation import evaluate_shared_worker, evaluate_agent_batched

# Anchor all paths to this script's directory so it works on any computer/IDE configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def warm_up_numba():
    """Wakes up the Numba JIT compiler before training begins."""
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


def evaluate_population_fast(agents, executor, games_per_agent=100, seed_offset=400000):
    """
    Evaluates the entire population of agents using Shared Memory and Numba
    across all available CPU cores with zero IPC serialization lag.
    """
    population_size = len(agents)

    genomes_list = [agent.network.get_genome() for agent in agents]
    genomes_arr = np.array(genomes_list, dtype=np.float64)

    shm = shared_memory.SharedMemory(create=True, size=genomes_arr.nbytes)
    shared_genomes = np.ndarray(genomes_arr.shape, dtype=genomes_arr.dtype, buffer=shm.buf)
    shared_genomes[:] = genomes_arr[:]

    tasks = [
        (shm.name, genomes_arr.shape, genomes_arr.dtype, "circular", games_per_agent, seed_offset, i)
        for i in range(population_size)
    ]

    try:
        workers = os.cpu_count() or 4
        results = list(
            executor.map(evaluate_shared_worker, tasks, chunksize=max(1, population_size // (workers * 4))))
    finally:
        shm.close()
        shm.unlink()

    return np.array(results)


def tournament_selection(agents, scores, tournament_size=3):
    selected_indices = np.random.choice(len(agents), size=tournament_size, replace=False)
    best_in_tourney = selected_indices[np.argmin(scores[selected_indices])]
    return agents[int(best_in_tourney)].network.get_genome()


def crossover(genome1, genome2):
    """Performs single-point crossover with strict length validation."""
    if len(genome1) != len(genome2):
        min_len = min(len(genome1), len(genome2))
        genome1 = genome1[:min_len]
        genome2 = genome2[:min_len]

    point = int(np.random.randint(1, len(genome1)))
    child1 = np.concatenate([genome1[:point], genome2[point:]])
    child2 = np.concatenate([genome2[:point], genome1[point:]])
    return child1, child2


def mutate(genome, mutation_rate=0.1, mutation_scale=0.2):
    """Applies Gaussian mutation safely to a genome."""
    mutated = genome.copy()
    mask = np.random.rand(len(mutated)) < mutation_rate
    mutations = np.random.normal(0, mutation_scale, size=len(mutated))
    mutated[mask] += mutations[mask]
    return mutated


def inspect_winning_strategy(best_agent, seed=42):
    """Runs a single game using the correct Game and Agent methods."""
    game = Game(seed)
    best_agent.reset(seed)

    print("\n========== WINNING AGENT STRATEGY INSPECTION ==========")

    turn = 0
    while not game.finished:
        if game.turns >= MAX_TURNS_PER_GAME:
            break

        turn += 1
        current_probs = best_agent.probabilities.copy()
        guess = best_agent.choose_guess()

        print(f"\n--- Turn {turn} ---")
        print(f"Current Probability Beliefs: {np.round(current_probs, 3)}")
        print(f"Network Guessed Box: {guess}")

        found = game.guess(guess)
        if found:
            print(f"-> SUCCESS! Cat found in box {guess} on turn {turn}.")
            break
        else:
            print(f"-> Missed. Beliefs updated.")
            best_agent.update(guess)

    print(f"Total guesses required: {game.turns}")
    print("=======================================================\n")


def main():
    print("Initializing evolutionary training pipeline...")
    warm_up_numba()

    MAX_RUNS = 20
    generations = 1000
    population_size = 100
    games_per_agent = 100
    workers = os.cpu_count() or 4

    elitism_count = 5
    tournament_size = 4
    base_mutation_rate = 0.15
    mutation_scale = 0.3

    # Spin up the worker pool ONCE for all automated background runs across batches
    with ProcessPoolExecutor(max_workers=workers) as executor:
        while True:
            # Automatically find the first incomplete batch and run slot using script-relative paths
            batch_idx = 1
            target_batch_dir = None
            target_run_idx = None

            while target_run_idx is None:
                batch_dir = os.path.join(BASE_DIR, f"batch_{batch_idx}")

                # Create the batch folder if it doesn't exist yet
                if not os.path.exists(batch_dir):
                    os.makedirs(batch_dir, exist_ok=True)
                    print(f"[Folder Created] Initialized new batch directory: batch_{batch_idx}/")

                run_idx = 1
                while run_idx <= MAX_RUNS and os.path.exists(os.path.join(batch_dir, f"run{run_idx}.npz")):
                    run_idx += 1

                if run_idx <= MAX_RUNS:
                    target_batch_dir = batch_dir
                    target_run_idx = run_idx
                else:
                    batch_idx += 1

            print(f"\n==================================================")
            print(f"   STARTING BATCH {batch_idx} | RUN {target_run_idx} OF {MAX_RUNS}")
            print(f"==================================================")

            # Completely reset population and tracking variables for this run
            agents: list[Agent] = [Agent(network_type="circular") for _ in range(population_size)]
            stagnation_counter = 0
            best_gen_score = float('inf')

            for gen in range(generations):
                # Unique seed offset per batch, run, and generation
                seed_offset = 400000 + (gen * 10000) + (batch_idx * 100000) + (target_run_idx * 1000000)

                start_time = time.perf_counter()
                scores = evaluate_population_fast(agents, executor, games_per_agent=games_per_agent,
                                                  seed_offset=seed_offset)
                elapsed = time.perf_counter() - start_time

                sorted_indices = np.argsort(scores)
                agents = [agents[int(i)] for i in sorted_indices]
                scores = scores[sorted_indices]

                best_score = scores[0]
                mean_score = np.mean(scores)

                print(
                    f"Batch {batch_idx} | Run {target_run_idx} | Gen {gen + 1}/{generations} | Best Score: {best_score:.3f} | Mean Score: {mean_score:.3f} | Eval Time: {elapsed:.3f}s")

                if best_score < best_gen_score:
                    best_gen_score = best_score
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                current_mutation_rate = base_mutation_rate
                if stagnation_counter > 3:
                    current_mutation_rate = base_mutation_rate * 2.0
                    print(
                        f"[Adaptive Mutation] Stagnation detected ({stagnation_counter} gen). Boosting mutation rate to {current_mutation_rate}")

                # Inspect winning strategy every 100 generations or on final generation
                if (gen + 1) % 100 == 0 or gen == generations - 1:
                    inspect_winning_strategy(agents[0], seed=500000 + gen)

                new_agents: list[Agent] = []

                for i in range(elitism_count):
                    elite_agent = Agent(network_type="circular")
                    elite_agent.network.set_genome(agents[i].network.get_genome())
                    new_agents.append(elite_agent)

                while len(new_agents) < population_size:
                    parent1_genome = tournament_selection(agents, scores, tournament_size)
                    parent2_genome = tournament_selection(agents, scores, tournament_size)

                    child1_genome, child2_genome = crossover(parent1_genome, parent2_genome)

                    child1_genome = mutate(child1_genome, mutation_rate=current_mutation_rate,
                                           mutation_scale=mutation_scale)
                    child2_genome = mutate(child2_genome, mutation_rate=current_mutation_rate,
                                           mutation_scale=mutation_scale)

                    for g in [child1_genome, child2_genome]:
                        if len(new_agents) < population_size:
                            child_agent = Agent(network_type="circular")
                            child_agent.network.set_genome(g)
                            new_agents.append(child_agent)

                agents = new_agents

            # 4. Robust Final Selection: Evaluate top 5 final agents over 1,000 games to find the true lowest average score
            print(
                f"\n[Batch {batch_idx} | Run {target_run_idx}] Training complete. Running rigorous 1,000-game validation on top candidates...")

            validation_games = 1000
            validation_seed = 900000 + (batch_idx * 50000) + (target_run_idx * 100000)

            best_verified_score = float('inf')
            best_verified_genome = None

            for rank, candidate in enumerate(agents[:5]):
                verified_avg = evaluate_agent_batched(candidate, games=validation_games, seed_offset=validation_seed)
                print(
                    f"  -> Candidate #{rank + 1} Verified Average Score ({validation_games} games): {verified_avg:.4f}")
                if verified_avg < best_verified_score:
                    best_verified_score = verified_avg
                    best_verified_genome = candidate.network.get_genome().copy()

            # Save the rigorously verified champion using the absolute script-relative path
            filename = os.path.join(target_batch_dir, f"run{target_run_idx}.npz")
            np.savez(filename, genome=best_verified_genome, score=best_verified_score)
            print(f"[Saved] Verified Champion saved to {filename} with True Average Score: {best_verified_score:.4f}\n")


if __name__ == "__main__":
    main()