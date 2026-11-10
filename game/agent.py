import random
import numpy as np

from config import BOX_COUNT
from ai.neural_network import NeuralNetwork
from ai.circular_neural_network import CircularNeuralNetwork

class Agent:

    def __init__(self, network_type="standard"):
        if network_type == "standard":
            self.network = NeuralNetwork()

        elif network_type == "circular":
            self.network = CircularNeuralNetwork()

        else:
            raise ValueError(f"Unknown network type: {network_type}")

        self.reset()

    def choose_guess_batch(self, probabilities, rng):
        """
        Selects guesses for a batch of probability distributions.
        probabilities shape: (num_games, BOX_COUNT)
        returns: array of guesses of shape (num_games,)
        """
        # Forward pass for the entire batch
        scores = self.network.forward(probabilities)

        # Mask impossible guesses
        scores[probabilities == 0] = -np.inf

        # Fully vectorized tie-breaking using tiny random noise
        noise = rng.uniform(0, 1e-5, size=scores.shape)
        guesses = np.argmax(scores + noise, axis=1)

        return guesses

    def update_batch(self, probabilities, guesses, miss_indices, rng):
        """
        Updates probabilities for a subset of games that missed their guess.
        """
        if len(miss_indices) == 0:
            return probabilities

        p_subset = probabilities[miss_indices]
        m_guesses = guesses[miss_indices]

        # 1. Zero out guessed box
        p_subset[np.arange(len(miss_indices)), m_guesses] = 0.0

        # 2. Renormalize remaining probabilities
        row_sums = np.sum(p_subset, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        p_subset /= row_sums

        # 3. Shift belief distributions to account for cat movement using np.roll
        probabilities[miss_indices] = 0.5 * (
                np.roll(p_subset, 1, axis=1) + np.roll(p_subset, -1, axis=1)
        )
        return probabilities
    def reset(self, seed=None):

        self.probabilities = np.full(
            BOX_COUNT,
            1 / BOX_COUNT
        )

        self.rng = random.Random(
            None if seed is None else seed + 123456789
        )
    def choose_guess(self):

        scores = self.network.forward(
            self.probabilities
        )

        # Do not guess positions where the cat cannot be
        scores[self.probabilities == 0] = -np.inf

        # Find the highest-scoring positions
        best_positions = np.where(
            scores == np.max(scores)
        )[0]

        return self.rng.choice(best_positions)

    def explain_strategy(self):

        scores = self.network.forward(self.probabilities)

        # Ignore impossible positions
        valid_scores = scores.copy()
        valid_scores[self.probabilities == 0] = -np.inf

        best_position = np.argmax(valid_scores)

        print("\n--- AI Strategy ---")

        print("Beliefs:", np.round(self.probabilities, 3))
        print("Network scores:", np.round(scores, 3))

        print("Chosen position:", best_position)

        print("Reasoning:")

        for i in range(BOX_COUNT):

            if self.probabilities[i] == 0:
                print(f"Box {i}: Impossible")
            else:
                print(
                    f"Box {i}: "
                    f"Probability = {self.probabilities[i]:.3f}, "
                    f"Score = {scores[i]:.3f}"
                )

    def update(self, guess):

        # Eliminate the position we just guessed
        self.probabilities[guess] = 0

        # Renormalise remaining probabilities
        self.probabilities /= np.sum(self.probabilities)

        # Account for the cat's movement
        new_probabilities = np.zeros(BOX_COUNT)

        for i in range(BOX_COUNT):
            new_probabilities[(i + 1) % BOX_COUNT] += (
                    self.probabilities[i] * 0.5
            )

            new_probabilities[(i - 1) % BOX_COUNT] += (
                    self.probabilities[i] * 0.5
            )

        self.probabilities = new_probabilities
