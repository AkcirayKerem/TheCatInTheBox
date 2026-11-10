import time
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
import numpy as np

from config import BOX_COUNT
from game.agent import Agent
from ai.evaluation import evaluate_agent_batched
from ai.circular_neural_network import _numba_conv_forward
from game.batch_game import _numba_step_kernel


def evaluate_shared_worker(args):
    shm_name, shape, dtype, network_type, games, seed_offset, agent_idx = args

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    genomes = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    genome = genomes[agent_idx].copy()
    existing_shm.close()

    agent = Agent(network_type=network_type)
    agent.network.set_genome(genome)

    return evaluate_agent_batched(agent, games=games, seed_offset=seed_offset + agent_idx)


def warm_up_numba():
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


def benchmark_shared_memory_population():
    print("\n========== STABLE SHARED MEMORY + NUMBA BENCHMARK ==========")

    population_size = 100
    games_per_agent = 100
    seed_offset = 400000

    print("[DEBUG] Warming up Numba JIT compiler...")
    warm_up_numba()

    agents = [
        Agent(network_type="circular")
        for _ in range(population_size)
    ]

    genomes_list = [agent.network.get_genome() for agent in agents]
    genomes_arr = np.array(genomes_list, dtype=np.float64)

    shm = shared_memory.SharedMemory(create=True, size=genomes_arr.nbytes)
    shared_genomes = np.ndarray(genomes_arr.shape, dtype=genomes_arr.dtype, buffer=shm.buf)
    shared_genomes[:] = genomes_arr[:]

    tasks = [
        (shm.name, genomes_arr.shape, genomes_arr.dtype, "circular", games_per_agent, seed_offset, i)
        for i in range(population_size)
    ]

    workers = os.cpu_count()
    print(f"Evaluating population of 100 agents using {workers} cores...")

    total_start = time.perf_counter()

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(
                executor.map(evaluate_shared_worker, tasks, chunksize=max(1, population_size // (workers * 4))))
    finally:
        shm.close()
        shm.unlink()

    total_elapsed = time.perf_counter() - total_start

    print(f"\n========== TIMING SUMMARY ==========")
    print(f"Total time: {total_elapsed:.3f}s")
    print(f"Results calculated: {len(results)}")
    print(f"Best agent average score: {np.min(results):.3f}")


if __name__ == "__main__":
    benchmark_shared_memory_population()