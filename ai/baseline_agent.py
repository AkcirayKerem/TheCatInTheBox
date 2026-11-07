import random
import numpy as np

from config import BOX_COUNT


class BaselineAgent:

    def reset(self, seed=None):
        self.probabilities = np.full(
            BOX_COUNT,
            1 / BOX_COUNT
        )

        self.rng = random.Random(
            None if seed is None else seed + 123456789
        )

    def choose_guess(self):
        best_positions = np.where(
            self.probabilities == np.max(self.probabilities)
        )[0]

        return self.rng.choice(best_positions)

    def update(self, guess):
        self.probabilities[guess] = 0
        self.probabilities /= np.sum(self.probabilities)

        new_probabilities = np.zeros(BOX_COUNT)

        for i in range(BOX_COUNT):
            new_probabilities[(i + 1) % BOX_COUNT] += (
                self.probabilities[i] * 0.5
            )

            new_probabilities[(i - 1) % BOX_COUNT] += (
                self.probabilities[i] * 0.5
            )

        self.probabilities = new_probabilities