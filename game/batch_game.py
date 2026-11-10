import numpy as np
from numba import njit
from config import BOX_COUNT, MAX_TURNS_PER_GAME


@njit(fastmath=True, cache=True)
def _numba_step_kernel(
        cat_positions, probabilities, turns, finished, guesses, rng_ints, rng_uniforms
):
    num_games = len(cat_positions)

    for i in range(num_games):
        if finished[i]:
            continue

        guess = guesses[i]
        cat_pos = cat_positions[i]
        turns[i] += 1

        if guess == cat_pos:
            finished[i] = True
        elif turns[i] >= MAX_TURNS_PER_GAME:
            finished[i] = True
        else:
            # Miss: Zero out guessed box probability
            probabilities[i, guess] = 0.0

            # Renormalize probabilities
            row_sum = 0.0
            for b in range(BOX_COUNT):
                row_sum += probabilities[i, b]

            if row_sum == 0.0:
                probabilities[i, :] = 1.0 / BOX_COUNT
            else:
                for b in range(BOX_COUNT):
                    probabilities[i, b] /= row_sum

            # Cat movement: randomly shift left (-1) or right (+1)
            # rng_ints[i] gives 1 or 2
            move_right = (rng_ints[i] == 1)
            if move_right:
                cat_positions[i] = (cat_positions[i] + 1) % BOX_COUNT
            else:
                cat_positions[i] = (cat_positions[i] - 1) % BOX_COUNT

            # Shift belief distributions (equivalent to roll by +1 and -1)
            old_p = probabilities[i].copy()
            for b in range(BOX_COUNT):
                left_val = old_p[(b - 1) % BOX_COUNT]
                right_val = old_p[(b + 1) % BOX_COUNT]
                probabilities[i, b] = 0.5 * (left_val + right_val)


class BatchGame:

    def __init__(self, num_games=100, seed=None):
        self.num_games = num_games
        self.rng = np.random.default_rng(seed)

        self.cat_positions = np.zeros(num_games, dtype=np.int64)
        self.probabilities = np.zeros((num_games, BOX_COUNT), dtype=np.float64)
        self.turns = np.zeros(num_games, dtype=np.float64)
        self.finished = np.zeros(num_games, dtype=np.bool_)

        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.cat_positions = self.rng.integers(0, BOX_COUNT, size=self.num_games, dtype=np.int64)
        self.probabilities.fill(1.0 / BOX_COUNT)
        self.turns.fill(0)
        self.finished.fill(False)

    def step(self, guesses):
        # Generate random inputs for Numba kernel
        rng_ints = self.rng.integers(1, 3, size=self.num_games, dtype=np.int64)

        _numba_step_kernel(
            self.cat_positions,
            self.probabilities,
            self.turns,
            self.finished,
            guesses.astype(np.int64),
            rng_ints,
            None
        )
        return self.finished.copy()

    @property
    def all_finished(self):
        return np.all(self.finished)

    def get_results(self):
        self.turns[~self.finished] = MAX_TURNS_PER_GAME
        return np.mean(self.turns)