import numpy as np
from numba import njit
from config import BOX_COUNT


@njit(fastmath=True, cache=True)
def _numba_conv_forward(inputs, weights, bias, conv_indices):
    """
    inputs: (batch_size, input_channels, BOX_COUNT)
    weights: (output_channels, input_channels, kernel_size)
    bias: (output_channels,)
    """
    batch_size, in_channels, _ = inputs.shape
    out_channels, _, kernel_size = weights.shape

    output = np.zeros((batch_size, out_channels, BOX_COUNT), dtype=np.float64)

    for b in range(batch_size):
        for o in range(out_channels):
            for n in range(BOX_COUNT):
                val = bias[o]
                for i in range(in_channels):
                    for k in range(kernel_size):
                        idx = conv_indices[n, k]
                        val += weights[o, i, k] * inputs[b, i, idx]
                output[b, o, n] = val

    return output


class CircularNeuralNetwork:

    def __init__(self, channels=8):
        self.channels = channels
        self.kernel_size = 3
        self.radius = self.kernel_size // 2

        # Precompute indices for zero-loop convolution
        n = np.arange(BOX_COUNT)
        k = np.arange(self.kernel_size)[:, None]
        # Shape: (BOX_COUNT, kernel_size)
        self.conv_indices = ((n[:, None] + k.T - self.radius) % BOX_COUNT).astype(np.int64)

        self.weights1 = (
                np.random.randn(channels, 1, self.kernel_size)
                * np.sqrt(1 / self.kernel_size)
        )
        self.bias1 = np.zeros(channels)

        self.weights2 = (
                np.random.randn(channels, channels, self.kernel_size)
                * np.sqrt(1 / (channels * self.kernel_size))
        )
        self.bias2 = np.zeros(channels)

        self.weights3 = (
                np.random.randn(channels)
                * np.sqrt(1 / channels)
        )
        self.bias3 = 0.0

    def circular_convolution(self, inputs, weights, bias):
        if inputs.ndim == 2:
            inputs = inputs[:, np.newaxis, :]

        out = _numba_conv_forward(inputs, weights, bias, self.conv_indices)
        return out

    def forward(self, inputs):
        x = np.asarray(inputs, dtype=float)
        is_single = (x.ndim == 1)
        if is_single:
            x = x.reshape(1, BOX_COUNT)

        # Layer 1
        x = self.circular_convolution(x, self.weights1, self.bias1)
        x = np.tanh(x)

        # Layer 2
        x = self.circular_convolution(x, self.weights2, self.bias2)
        x = np.tanh(x)

        # Output layer
        outputs = np.einsum('c,bcn->bn', self.weights3, x) + self.bias3

        return outputs.flatten() if is_single else outputs

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

        self.weights1 = genome[i:i + size1].reshape(self.channels, 1, self.kernel_size)
        i += size1
        self.bias1 = genome[i:i + size2]
        i += size2
        self.weights2 = genome[i:i + size3].reshape(self.channels, self.channels, self.kernel_size)
        i += size3
        self.bias2 = genome[i:i + size4]
        i += size4
        self.weights3 = genome[i:i + size5]
        i += size5
        self.bias3 = genome[i]