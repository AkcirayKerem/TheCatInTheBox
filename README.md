# TheCatInTheBox

An evolutionary AI project exploring neural networks, probabilistic decision-making, and optimisation through a circular search game.

## Overview

TheCatInTheBox is a Python-based project in which neural networks learn to solve a probabilistic search problem using a genetic algorithm.

The objective is to develop an AI capable of finding a hidden cat in a circular arrangement of boxes while minimising the average number of guesses required.

Rather than manually programming an optimal strategy, this project investigates whether evolutionary algorithms can discover increasingly effective strategies through repeated evaluation, selection, crossover, and mutation.

## The Game

The current implementation uses 10 boxes arranged in a circle.

The rules are:

1. The cat starts in a uniformly random box.
2. The AI selects a box to search.
3. If the cat is found, the game ends.
4. Otherwise, the cat moves exactly one position clockwise or anticlockwise, each with a probability of 50%.
5. The AI repeats the process until the cat is found.

The objective is to minimise the expected number of guesses.

## AI Architecture

Two neural network architectures are currently implemented:

- **Standard Neural Network:** A conventional fully connected neural network.
- **Circular Neural Network:** A neural network using circular convolutions to exploit the rotational symmetry of the game.

The circular architecture is particularly interesting because rotating the input probability distribution should produce an equivalently rotated output.

## Evolutionary Optimisation

The neural networks are optimised using a genetic algorithm.

The evolutionary process includes:

- Population-based evolution.
- Fitness evaluation through simulated games.
- Tournament selection.
- Elitism.
- Genetic crossover.
- Random mutation.
- Adaptive mutation in response to stagnation.
- Successive evaluation to reduce computational costs.

The fitness function is based on the average number of guesses required to find the cat.

## Baseline Comparison

A separate baseline agent uses a greedy maximum-probability strategy: it always selects the box with the highest current probability of containing the cat.

This provides a reference point against which evolved neural networks can be evaluated.

Preliminary experiments have shown that the evolved neural network can outperform this baseline, although further testing is needed to establish the reliability and significance of the improvement.

## Current Objectives

The primary objective is to improve the evolutionary training process and investigate how effectively neural networks can learn near-optimal strategies.

Planned development:

- [ ] Improve evolutionary training efficiency and reliability.
- [ ] Optimise game simulation and evaluation performance.
- [ ] Compare neural network architectures.
- [ ] Establish the mathematically optimal strategy for comparison.
- [ ] Investigate whether evolution can discover globally optimal or near-optimal solutions.
- [ ] Expand the game to support configurable numbers of boxes.
- [ ] Develop a GUI for configuring and running experiments.
- [ ] Visualise the optimisation landscape, including local and global minima.
- [ ] Investigate how game complexity affects the effectiveness of evolutionary optimisation.

## Technology

- Python
- NumPy
- Genetic Algorithms
- Neural Networks
- Circular Convolution
- Multiprocessing

Additional optimisation technologies, such as Numba, may be explored.

## Project Structure

```text
TheCatInTheBox/
│
├── ai/
│   ├── baseline_agent.py
│   ├── circular_neural_network.py
│   ├── neural_network.py
│   ├── evaluation.py
│   ├── genetic_algorithm.py
│   ├── benchmark.py
│   └── ai_debug_circular.py
│
├── game/
│   ├── agent.py
│   └── game.py
│
├── config.py
├── main.py
└── README.md
```

Long-Term Vision

The ultimate goal is to develop an interactive GUI through which users can configure the number of boxes and observe how the evolutionary AI performs under different configurations.

A central feature will be a 3D visualisation of the optimisation landscape, examining the relationship between:

Number of boxes: How the size of the search environment affects the problem.

Local minima: Suboptimal solutions discovered during optimisation.

Global minimum: The lowest theoretically achievable average number of guesses.

Mathematical trends: How the optimal solution and optimisation landscape change as the number of boxes increases.

The intention is to investigate whether these relationships follow identifiable mathematical patterns, and potentially derive a mathematical model describing how the optimal expected number of guesses scales with the number of boxes.

Ultimately, the project aims to combine evolutionary computation, mathematical analysis, and interactive visualisation to better understand the structure and complexity of this probabilistic search problem.
