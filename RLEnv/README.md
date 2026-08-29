# Continuous Control Problems using RL

This repository contains the source code and implementations for the project titled **Optimal Control of Continuous Time Systems using RL**, for the course RO3004: Reinforcement Learning Fundamentals at Plaksha University (Spring 2026).

# Setup

The `conda` package manager is preferred to setup the python environment using the `environment.yml` file.


## Conda:
```
conda env create -f environment.yml -n <env_name>
```

# File Structure
- `analytical_scripts/`: MATLAB scripts to derive expressions, and test numerically computed variables in python.
- `PIRL/`: Implementation of the PIRL algorithm from [1].
    - `config.yaml`: Environment and hyperparameter config (training configued within `train.py`)
    - `main.py`: Calling scirpt
    - `NeuralNets.py`: Definitions of actor-critic networks and variants.
    - `utils.py`: State sampler, and other utils.
    - `PIRL.py`: Definitions of the 'PIRL' agents
    - `train.py`: Implementations of the 'PIRL' algorithms
        - `Algo1`: Model based policy iteration (solves HJB PDE)
        - `Algo2`: Model-free policy iteration (solves integral HJB equation)
- `training/`: Utilitiea and function definitions used throughout the codebase.
    - `callbacks.py`: stable-baselines3 callback definitions
    - `disturbances.py`, `rewards.py`: Disturbance and reward function definitions
    - `evaluation.py`: Run rollouts and make plots
- `baseenv.py`: gym inhertited base class for our environments
- `invertedpendulum.py`: inverted pendulum environment 
- `experiment.py`: Calling script to run stable-baselines3 algorithms

# Reproduce
If the python environment is correctly setup, `PIRL/main.py` for PIRL algorithms and `experiment.py` for SB3 algorithms are the main calling scripts which setup the gym environment, train, and log / plot results. 


