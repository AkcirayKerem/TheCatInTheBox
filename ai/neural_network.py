import numpy as np

from config import BOX_COUNT, HIDDEN_SIZE
class NeuralNetwork:

    def __init__(self):

        self.weights1 = (
            np.random.randn(BOX_COUNT, HIDDEN_SIZE) * 0.5
        )

        self.bias1 = np.zeros(HIDDEN_SIZE)

        self.weights2 = (
            np.random.randn(HIDDEN_SIZE, BOX_COUNT) * 0.5
        )

        self.bias2 = np.zeros(BOX_COUNT)

    def forward(self, inputs):

        hidden = np.tanh(
            np.dot(inputs, self.weights1) + self.bias1
        )

        outputs = (
            np.dot(hidden, self.weights2) + self.bias2
        )

        return outputs

    def get_genome(self):

        return np.concatenate([
            self.weights1.flatten(),
            self.bias1.flatten(),
            self.weights2.flatten(),
            self.bias2.flatten()
        ])

    def set_genome(self, genome):

        i = 0

        size1 = BOX_COUNT * HIDDEN_SIZE
        size2 = HIDDEN_SIZE
        size3 = HIDDEN_SIZE * BOX_COUNT
        size4 = BOX_COUNT

        self.weights1 = genome[i:i + size1].reshape(
            BOX_COUNT, HIDDEN_SIZE
        )
        i += size1

        self.bias1 = genome[i:i + size2]
        i += size2

        self.weights2 = genome[i:i + size3].reshape(
            HIDDEN_SIZE, BOX_COUNT
        )
        i += size3

        self.bias2 = genome[i:i + size4]