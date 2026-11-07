
import numpy as np
from config import BOX_COUNT


class CircularNeuralNetwork:

    def __init__(self, channels=8):
        self.channels = channels
        self.kernel_size = 3

        # First convolution: 1 input channel -> channels
        self.weights1 = (
            np.random.randn(channels, 1, self.kernel_size)
            * np.sqrt(1 / self.kernel_size)
        )
        self.bias1 = np.zeros(channels)

        # Second convolution: channels -> channels
        self.weights2 = (
            np.random.randn(channels, channels, self.kernel_size)
            * np.sqrt(1 / (channels * self.kernel_size))
        )
        self.bias2 = np.zeros(channels)

        # Shared output layer: channels -> 1 score per position
        self.weights3 = (
            np.random.randn(channels)
            * np.sqrt(1 / channels)
        )
        self.bias3 = 0.0

    def circular_convolution(self, inputs, weights, bias):
        """
        inputs:  (input_channels, BOX_COUNT)
        weights: (output_channels, input_channels, kernel_size)
        """
        output_channels, input_channels, kernel_size = weights.shape
        radius = kernel_size // 2

        # Vectorize the rolling across all kernel positions at once (no Python loop)
        shifted = np.stack([
            np.roll(inputs, shift=-(k - radius), axis=1)
            for k in range(kernel_size)
        ], axis=0)  # Shape: (kernel_size, input_channels, BOX_COUNT)

        # Single optimized C-level contraction
        output = np.einsum('oik,kin->on', weights, shifted)
        output += bias[:, None]

        return output

    def forward(self, inputs):
        x = np.asarray(inputs, dtype=float).reshape(1, BOX_COUNT)

        # Convolutional layer 1
        x = self.circular_convolution(
            x,
            self.weights1,
            self.bias1
        )
        x = np.tanh(x)

        # Convolutional layer 2
        x = self.circular_convolution(
            x,
            self.weights2,
            self.bias2
        )
        x = np.tanh(x)

        # Shared output layer
        outputs = np.dot(self.weights3, x) + self.bias3

        return outputs.flatten()

    def get_genome(self):
        return np.concatenate([
            self.weights1.flatten(),
            self.bias1.flatten(),
            self.weights2.flatten(),
            self.bias2.flatten(),
            self.weights3.flatten(),
            np.array([self.bias3])
        ])

    def set_genome(self, genome):
        i = 0

        size1 = self.channels * self.kernel_size
        size2 = self.channels
        size3 = self.channels * self.channels * self.kernel_size
        size4 = self.channels
        size5 = self.channels

        self.weights1 = genome[i:i + size1].reshape(
            self.channels, 1, self.kernel_size
        )
        i += size1

        self.bias1 = genome[i:i + size2]
        i += size2

        self.weights2 = genome[i:i + size3].reshape(
            self.channels, self.channels, self.kernel_size
        )
        i += size3

        self.bias2 = genome[i:i + size4]
        i += size4

        self.weights3 = genome[i:i + size5]
        i += size5

        self.bias3 = genome[i]
