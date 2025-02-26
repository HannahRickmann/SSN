# Primal-Dual Active Set Algorithm for Quadratic Optimization Problems

This repository includes an implementation of the Primal-Dual Active Set Algorithm (PDASA) for solving quadratic optimization problems, specifically focusing on scenarios with both lower and upper constraints. The theory behind the algorithm is based on the work of Ito and Kunisch (2008).
This implementation is the basis of the numerical examples in my master thesis 'Global Convergence Theory for nonsmooth Newton Methods applied to Quadratic Programs'.

It contains Python scripts for running various experiments related to constrained quadratic programming (QP). The provided code allows for testing different data configurations, analyzing convergence behavior, and evaluating active set cycles.

## Running Experiments

The main script to run experiments is main.py. The following functions perform different types of experiments:

### 1. Trying All Possible Data Combinations

Run all pre-constructed data configurations where each number satisfies $|x| < 10$.

```python
try_all_possibilities()
```

- Loads pre-generated data or constructs new data if unavailable.
- Iterates through all possible data combinations and runs the experiment.

### 2. Testing Scaling Effects

Multiply the problem data by a constant and compare results:

```python
try_multiplying_constant(exp_nr=17, c=-1)
```

- Runs an experiment for a given dataset (``exp_nr``).
- Experiment data is saved in ``./experiments/input/custom_experiment_nr_(exp_nr).json``.
- Multiplies the matrix $A$ and vector $b$ by $c$ and reruns the experiment.

### 3. Trying Different Upper Bounds

Test different upper bounds for convergence behavior:

```python
try_different_bounds(exp_nr=31)
```

- Runs the experiment with various upper bound values.
- Saves bounds that result in non-convergence.

### 4. Testing Active Set Cycles

Analyze possible active set cycles in the problem:

```python
try_possible_active_set_cycles(n=3, n_cycle=5)
```

- Tests active set cycles for a given dimension ``n`` and cycle count ``n_cycle``.
- Prints cycles to terminal, which fulfill necessary cycle behavior

### 5. Running Random Experiments

Generate and run random experiments of dimension n:

```python
try_random_experiment(amount=10000, n=7, lower_bound=False)
```

- Generates and solves ``amount`` number of random QP problems of size ``n``.
- Optionally includes lower bounds in the problem formulation.
- Saves examples with failed convergence cases in JSON files under ``./experiments/results/{current_time_stamp}``.

### 6. Re-running Previous Random Experiments

Rerun and analyze previously generated random experiments:

```python
try_previous_random_experiment(time='2024-12-12_16-43-36')
```

- Loads experiments saved at a specific timestamp.
- Analyzes the active set behavior and cycles of the experiments.

### 7. Running Custom Experiments

Run a custom experiment and optionally analyze it:

```python
try_custom_experiment(exp_nr=17, analyse=True)
```

- Loads a specific experiment (``exp_nr``) and runs it.
- If ``analyse=True``, it examines active sets and cycles.

## Obstacle Problem Solver

The script ``obstacle_problem.py`` contains a brief analysis of obstacle problems in one and two dimensions:

- **1D Obstacle Problem**: The 1D case considers a line mesh with constraints imposed by obstacle functions. The solution is computed using quadratic programming and visualized through plots.
- **2D Obstacle Problem**: The 2D case extends the analysis to a triangular mesh, incorporating obstacles as height constraints. A bilinear weak form of the PDE is used, and the problem is solved numerically. The solution is visualized in 2D and 3D representations.

The code also includes numerical validation of the convergence properties of the PDASA algorithm for solving the obstacle problem based on Ito and Kunisch (2008). 

## Reference
Ito, Kazufumi and Karl Kunisch (2008). Lagrange Multiplier Approach to Variational Problems and Applications. Advances in Design
and Control. Society for Industrial and Applied Mathematics. doi: 10.1137/1.9780898718614.
