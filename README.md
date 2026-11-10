# TheCatInTheBox

**An evolutionary AI and mathematical optimisation project exploring probabilistic search.**

TheCatInTheBox is a Python project that investigates how effectively an AI can locate a moving hidden object in a circular arrangement of boxes.

The project combines circular neural networks, genetic algorithms, probabilistic reasoning, accelerated simulation, and large-scale benchmarking to develop and compare search strategies.

Beyond finding effective strategies, the long-term objective is to investigate the mathematical limits of the problem, understand how optimal performance scales with the number of boxes, and visualise the resulting optimisation landscape.

## 1. The Game

The game consists of a circular arrangement of \(N\) boxes.

- The cat initially occupies a randomly selected box, with every box equally likely.
- The player selects a box in an attempt to find the cat.
- If the guess is incorrect, the cat moves exactly one position clockwise or anticlockwise, each with probability \(1/2\).
- The process continues until the cat is found.

The primary objective is to minimise the expected number of guesses required to find the cat.

The number of boxes is configurable through the project configuration, allowing experiments with different game sizes.

## 2. AI Architecture

The project uses neural networks trained through evolutionary optimisation rather than conventional gradient-based training.

### Circular Neural Network

The main AI architecture is a circular neural network designed around the rotational symmetry of the game.

Since every box has an equivalent structural position on the circle, the architecture uses circular convolution to process the game state while respecting this symmetry.

The network evaluates the possible next guesses, while the agent maintains and updates its probability distribution over the cat's possible locations.

Numba-compiled operations are used to accelerate computationally intensive neural-network calculations.

### Agent and Probabilistic Reasoning

The agent combines its neural-network output with information about which boxes have already been eliminated.

Following an incorrect guess, it updates its probability distribution to account for the cat's possible movement.

This allows the agent to base subsequent decisions on the information accumulated throughout the game.

### Baseline Strategy

The project also includes a greedy baseline agent.

This provides a reference point for assessing whether evolutionary training produces strategies that outperform a simpler search approach.

## 3. Evolutionary Training

The genetic algorithm evolves neural-network genomes through population-based optimisation.

Its principal mechanisms include:

- Population-based evaluation.
- Selection and elitism.
- Genome crossover.
- Genome mutation.
- Performance-based ranking.
- Adaptive mutation behaviour where applicable.

The evolutionary objective is to discover strategies that minimise average guesses across repeated games.

Training prioritises reliable average performance rather than placing excessive emphasis on isolated, unusually good generation results.

### Genome Safeguards

Additional validation has been introduced into crossover and mutation operations to check genome compatibility and prevent shape mismatches or accidental truncation.

NumPy index handling has also been improved through explicit conversion to standard Python integers.

## 4. Automated Training and Experiment Management

The training pipeline has been extended to support automated sequential experiments.

### Multi-Run Automation

The main training script automatically scans existing checkpoint files and identifies the next available run number.

Experiments are organised into batch directories:

- Each batch accommodates up to 20 training runs.
- Once a batch reaches its limit, a new batch directory is created automatically.
- Checkpoints follow the naming convention `runX.npz`.
- Additional batches can be created as further experiments are performed.

This allows repeated experiments to be organised without manually managing run numbers or directories.

### Training Cycles

Training is organised into cycles of up to 1,000 generations.

At the end of a cycle, the system does not simply trust the best score recorded during evolution. Instead, it performs a separate candidate-verification stage.

### Final Champion Verification

The five leading candidate agents undergo an additional evaluation consisting of 1,000 games using a fixed seed configuration.

Their average scores are compared, and the candidate with the lowest verified average number of guesses is selected as the champion.

The selected agent's genome is then saved as a checkpoint for subsequent analysis and comparison.

## 5. Evaluation and Reproducibility

Reliable evaluation is particularly important because small performance differences can significantly influence evolutionary rankings.

Several improvements have been introduced to make comparisons more consistent.

### Batched Game Evaluation

The project includes batched simulation through `batch_game.py`, allowing multiple games to be processed together.

Numba-compiled simulation kernels are used to accelerate game-state updates and reduce the overhead associated with repeatedly executing individual Python-level game operations.

The evaluation module also provides batched agent evaluation.

### Deterministic Tie-Breaking

A deterministic tie-breaking mechanism has been introduced into evaluation.

A fixed pseudo-random evaluation stream (`eval_rng`) is used to resolve tied decisions consistently, reducing unnecessary variation in leaderboard rankings.

This is particularly useful when comparing candidates whose performance is very close.

### Fixed-Seed Evaluation

Fixed seed configurations are used during candidate verification so that repeated evaluations can be compared consistently.

This supports reproducible comparisons, although independent test sets are still necessary when estimating performance on previously unseen games.

## 6. Multi-Run Champion Tournament

The project includes a dedicated tournament system, implemented in `tournament.py`, to compare champions from different evolutionary training runs.

Rather than evaluating each training run in isolation, the tournament gathers saved champion genomes across experiment batches and evaluates them under a common benchmarking procedure.

### Automatic Champion Discovery

The tournament scans the batch directory structure and identifies available champion checkpoints.

All discovered champions are included in the comparison, allowing agents from separate experiments to compete on the same evaluation framework.

### Multi-Trial Evaluation

The current tournament configuration uses:

- **5 independent evaluation trials.**
- **10,000 games per trial.**
- **50,000 total evaluation games per champion.**

Each trial generates a new seed offset. The same seed offset is supplied to every champion during that trial, supporting controlled comparisons between their performance on corresponding game sequences.

### Parallel Processing

Champion evaluations are distributed across CPU worker processes using `ProcessPoolExecutor`.

The system uses shared memory to make loaded champion genomes accessible to the worker processes without repeatedly transferring the full genome arrays.

Batched evaluation and Numba-compiled operations are used to improve simulation throughput.

A Numba warm-up routine is also called before the tournament begins.

### Baseline Comparison

The greedy baseline agent is evaluated during each trial and included in the final leaderboard.

The baseline evaluation currently uses a sequential game loop, while champion evaluations are distributed across worker processes.

### Tournament Leaderboard

After all trials are complete, the average score across the trials is calculated for each agent.

The tournament then sorts all champions and the baseline by average number of guesses, producing a comparative leaderboard.

This provides a consistent method for identifying which evolutionary training run has produced the strongest champion.

## 7. Project Structure

The main project components are organised into separate modules.

**AI components**

- `ai/circular_neural_network.py` — Circular neural-network architecture and accelerated convolution operations.
- `ai/genetic_algorithm.py` — Evolutionary training and genome optimisation.
- `ai/evaluation.py` — Agent evaluation and batched evaluation procedures.
- `ai/baseline_agent.py` — Greedy baseline strategy.
- `ai/benchmark.py` — Performance benchmarking.

**Game components**

- `game/game.py` — Original game implementation.
- `game/batch_game.py` — Batched game simulation.
- `game/agent.py` — Agent behaviour, probability updates, and decision-making.

**Main scripts and configuration**

- `config.py` — Game and simulation configuration.
- `main.py` — Automated training and sequential experiment management.
- `tournament.py` — Cross-run champion comparison and benchmarking.

Generated checkpoints and experiment data are organised separately from the main source code.


### Matrix-Based Computation

The project uses NumPy arrays and multidimensional matrix representations to process neural-network parameters and game states.

These representations support:

- Storage and manipulation of neural-network weights and biases.
- Circular convolution across the box positions.
- Batched probability distributions for simultaneous game simulations.
- Vectorised operations on multiple game states.
- Efficient storage and transfer of populations of neural-network genomes.

Combined with Numba-compiled operations, this numerical architecture reduces reliance on repeated Python-level calculations and improves the efficiency of evolutionary evaluation.
## 8. Running the Project

The project uses Python, NumPy, and Numba, alongside Python's standard multiprocessing and numerical libraries.

From the project root, activate the Python virtual environment and install the required dependencies.

Run the automated training pipeline:

`python main.py`

Run the cross-run champion tournament:

`python tournament.py`

The tournament requires previously generated champion checkpoint files in the batch directories.

## 9. Future Research and Development

The long-term objective is to combine evolutionary computation, mathematical optimisation, and interactive visualisation.

### Mathematical Optimal Solver

An independent mathematical solver is planned to calculate the lowest theoretically achievable expected number of guesses.

The intended approach is dynamic programming based on the Bellman optimality principle, supported by techniques such as:

- State-space reduction.
- Memoization.
- Exploitation of circular symmetry.
- Branch-and-bound optimisation.
- Mathematical upper and lower bounds.

The solver will provide an independent theoretical benchmark against which the genetic algorithm's performance can be compared.

### Mathematical Scaling Analysis

An important research objective is to investigate how the globally optimal expected number of guesses changes with the number of boxes.

The aim is to identify mathematical trends and, where possible, derive a relationship describing the scaling behaviour of the optimal solution.

Rather than assuming a particular mathematical function, the project will investigate candidate models using results obtained from the independent solver.

### Optimisation Landscape

Another objective is to investigate the structure of the optimisation landscape, including local minima, global minima, and the behaviour of evolutionary search.

This will require defining meaningful strategy parameters or a suitable lower-dimensional representation of the neural-network search space.

### Interactive GUI and 3D Visualisation

The eventual goal is to develop an interactive graphical interface that allows users to configure the number of boxes and explore the mathematical and computational behaviour of the game.

Planned features include:

- Configurable game sizes.
- Three-dimensional optimisation landscape visualisation.
- Identification and comparison of local and global minima.
- Comparison between evolved strategies and mathematically established optima.
- Investigation of mathematical trends as the number of boxes increases.

The exact variables used for the three-dimensional visualisations will be determined through the mathematical investigation.

## 10. Current Development Priorities

The immediate priorities are:

1. Develop an independent mathematical solver.
2. Establish rigorous optimality bounds for smaller game configurations.
3. Compare evolutionary performance against the theoretical optimum.
4. Investigate the mathematical scaling behaviour of the problem.
5. Develop the interactive visualisation and GUI.

The broader ambition is to move beyond simply finding an effective AI strategy and towards understanding the mathematical limits, optimisation structure, and scaling behaviour of the probabilistic search problem.