# Reservoir Computing for One-Shot Reinforcement Learning

## Overview

This repository contains a neuroscience-inspired implementation of reservoir computing models for efficient reinforcement learning. The project explores various approaches to one-shot learning in navigation tasks, comparing traditional gradient-based methods with reservoir-based approaches. The implementation focuses on how reservoir computing can facilitate fast adaptation and learning from limited examples through meta-learning.

## Repository Structure

### Backbone Files
- **`reservoir.py`**: Core implementation of the Reservoir Computing model, featuring a recurrent neural network with stochastic spiking units.
- **`agent.py`**: Implementation of the LinearAgent that uses policy gradients for learning.
- **`environment.py`**: Grid-based navigation environment where an agent must find food across different angular positions.

### Low-level Training Utilities
- **`trainingUtils.py`**: Core training functions including episode execution, in-distribution training, and out-of-distribution evaluation.
- **`reservoirTrainingUtils.py`**: Specialized utilities for training reservoirs with gradient information and performing inference.
- **`stagePredictorReservoir.py`**: Implements meta-learning mechanisms and state prediction using reservoir dynamics.

### High-level Simulation Scripts
- **`oneShotGradient.py`**: Evaluates one-shot gradient learning across various initial orientations.
- **`oneShotReservoirGradient.py`**: Uses a reservoir to predict policy gradients for one-shot learning.
- **`oneShotReservoirMultiplier.py`**: Explores the effect of learning rate multipliers when using reservoir-predicted gradients.
- **`trueOneShotReservoir.py`**: Implements true one-shot meta-learning using a trained reservoir to predict weight updates.
- **`trueOneShotReservoirMultiplier.py`**: Extends meta-learning with learning rate multiplication to study scaling effects.

### Plotting and Analysis
- **`entropyModulation.py`**: Implements entropy-based modulation to improve reservoir learning performance.
- **`plottingUtils.py`**: Provides standard plotting functions for reward curves, trajectories, and performance metrics.
- **`paperReproductionUtils.py`**: Utilities for reproducing experimental results from reference papers.
- **`presentationPlottingUtils.py`**: Advanced visualization tools for creating presentation-quality figures.

## Key Experiments

The repository implements several key experimental paradigms:

1. **Traditional Gradient Training**: Using policy gradient methods to train agents from scratch.
2. **One-Shot Reservoir Gradient**: Using a reservoir to predict policy gradients after a single reward.
3. **One-Shot Reservoir with Multipliers**: Exploring how scaling gradients affects learning performance.
4. **Meta-Learning with Reservoirs**: Training reservoirs to predict entire weight updates rather than just gradients.
5. **Entropy Modulation**: Testing how encoding policy entropy in the reservoir affects learning performance.

## Evaluation Methods

The codebase includes comprehensive evaluation methods:

- **In-Distribution Testing**: Performance on orientations used during training (multiples of 45°).
- **Out-of-Distribution Testing**: Generalization to novel orientations (multiples of 22.5°).
- **Comparative Analysis**: Direct comparison between gradient-based methods and reservoir-based approaches.

## Visualization

The repository provides extensive visualization tools for:
- Reward curves across training episodes
- Agent trajectories in the environment
- Performance across different initial orientations
- Comparative analysis across different methods

## Requirements

The code requires standard scientific Python libraries including NumPy, Matplotlib, and Seaborn. The implementation uses multiprocessing for parallel evaluation of different experimental conditions.

## Usage

Most experimental scripts can be executed directly. For example:

```powershell
python trueOneShotReservoir.py
```

This will run the meta-learning experiment with entropy modulation, save the results to a JSON file, and generate visualization plots.
