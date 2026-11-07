
import os
import time
import numpy as np

from concurrent.futures import ProcessPoolExecutor

from config import BOX_COUNT
from game.agent import Agent
from ai.baseline_agent import BaselineAgent
from ai.evaluation import evaluate_agent, evaluate_agent_worker


def benchmark_inference():
    print("\n========== NEURAL NETWORK INFERENCE ==========")

    agent = Agent(network_type="circular")
    probabilities = np.full(BOX_COUNT, 1 / BOX_COUNT)

    iterations = 10000

    # Warm-up
    for _ in range(100):
        agent.network.forward(probabilities)

    start = time.perf_counter()

    for _ in range(iterations):
        agent.network.forward(probabilities)

    elapsed = time.perf_counter() - start

    print(f"Iterations: {iterations}")
    print(f"Total time: {elapsed:.4f}s")
    print(f"Average: {elapsed / iterations * 1000:.6f} ms")
    print(f"Throughput: {iterations / elapsed:.0f} inferences/s")


def benchmark_baseline():
    print("\n========== BASELINE SIMULATION ==========")

    games = 1000

    start = time.perf_counter()

    average = evaluate_agent(
        BaselineAgent(),
        games=games,
        seed_offset=300000
    )

    elapsed = time.perf_counter() - start

    print(f"Games: {games}")
    print(f"Average guesses: {average:.3f}")
    print(f"Total time: {elapsed:.4f}s")
    print(f"Time per game: {elapsed / games:.6f}s")




def benchmark_evaluation():
    print("\n========== REALISTIC POPULATION SCALING ==========")

    population_size = 100
    games_per_agent = 10
    seed_offset = 400000

    agents = [
        Agent(network_type="circular")
        for _ in range(population_size)
    ]

    tasks = [
        (agent, games_per_agent, seed_offset)
        for agent in agents
    ]

    worker_counts = [1, 2, 4, 6, 8, 10, 12]

    measurements = {}

    for workers in worker_counts:

        print(f"\nTesting {workers} worker(s)...")

        start = time.perf_counter()

        with ProcessPoolExecutor(
            max_workers=workers
        ) as executor:

            results = list(
                executor.map(evaluate_agent_worker, tasks)
            )

        elapsed = time.perf_counter() - start

        measurements[workers] = elapsed

        print(f"Time: {elapsed:.3f}s")
        print(f"Results calculated: {len(results)}")

    baseline_time = measurements[1]

    print("\n========== FINAL COMPARISON ==========")

    for workers, elapsed in measurements.items():

        print(
            f"{workers} workers: "
            f"{elapsed:.3f}s | "
            f"Speedup: {baseline_time / elapsed:.2f}x"
        )


if __name__ == "__main__":
    benchmark_evaluation()


