import numpy as np
from config import MAX_TURNS_PER_GAME
from game.batch_game import BatchGame
from game.agent import Agent


def evaluate_agent_batched(agent, games=100, seed_offset=0):
    game = BatchGame(num_games=games, seed=seed_offset)

    # Create a dedicated local RNG seeded with the offset
    # to guarantee 100% reproducible tie-breaking across tournament runs.
    eval_rng = np.random.default_rng(seed_offset + 888888)

    while not game.all_finished:
        inputs = game.probabilities
        scores = agent.network.forward(inputs)

        if scores.ndim > 1:
            masked_scores = scores.copy()
            masked_scores[game.probabilities == 0.0] = -1e9

            # Add tiny reproducible noise to cleanly break ties
            # without altering true neural network strategy preferences.
            noise = eval_rng.uniform(0, 1e-5, size=masked_scores.shape)
            guess_indices = np.argmax(masked_scores + noise, axis=1)
        else:
            guess_indices = np.argmax(scores)

        game.step(guess_indices)

    return game.get_results()


def evaluate_shared_worker(args):
    from multiprocessing import shared_memory
    shm_name, shape, dtype, network_type, games, seed_offset, agent_idx = args

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    genomes = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    genome = genomes[agent_idx].copy()
    existing_shm.close()

    agent = Agent(network_type=network_type)
    agent.network.set_genome(genome)

    return evaluate_agent_batched(agent, games=games, seed_offset=seed_offset + agent_idx)