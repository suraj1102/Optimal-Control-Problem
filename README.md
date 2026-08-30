# Optimal Control of Continuous-Time Systems

This repository contains the source code for the course projects of AI3001: Deep Learning (Monsoon 2025) and RO3004: Reinforcement Learning, Plaksha University (Spring 2026) at Plaksha University. Team: Arnav Kapoor, Meher Sidhu, Suraj Dayma.

Description: Using physics-informed neural networks (PINNs) to solve optimal control problems like inverted pendulum and cart pole by solving the Hamilton-Jacobi-Bellman (HJB) PDE. For the RL course, this was extended to solve the inverted pendulum problem using different RL algorithms and comparing the PINN based approach and RL policies with a trajectory optimisation benchmark for both constrained and unconstrained control cases.   

## Problem Statement

The Hamilton–Jacobi–Bellman PDE gives the optimal value function $V^*$, and for
control-affine dynamics its gradient hands you the optimal feedback law for free:

$$u^* = -\tfrac{1}{2} R^{-1} g(x)^\top \nabla V(x)$$

So is it better to **solve that PDE directly** with a physics-informed neural
network, or to **learn** the same thing with reinforcement learning?

## Result

For a control-affine system with known dynamics, a PINN (X-TFC architecture, see [this paper](https://ieeexplore.ieee.org/document/11372426) for details) is easier to set up, faster to train, and more accurate than physics-informed RL ([paper](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.18542)) and most model-free baselines. SAC is the best of the standard RL algorithms.

Steady-state error, mean $|\theta|$ over the last 100 steps of a 1000-step
rollout from $x_0 = [0.1, 1.0]$:

| Method | Type | $\|\theta\|$ (rad) |
|---|---|---|
| Trajectory optimization | benchmark (open-loop) | 0.0000 |
| SAC | model-free RL | 0.0066 |
| X-TFC (direct HJB) | physics-informed PINN | 0.0111 |
| DDPG | model-free RL | 0.0169 |
| PIRL-1 | physics-informed RL | 0.0532 |
| TD3 | model-free RL | 0.0974 |

There is also a clean negative result. PIRL-2, the model-free variant, cannot
work on this system: it needs an initial policy certified by a *quadratic*
control Lyapunov function, and no quadratic bowl decreases along the flow around
an unstable equilibrium. The same code converges fine on an inherently stable
system, which is how we confirmed the cause.

## Layout

Two halves sharing one repo.

**Repo root — direct HJB solving with PINNs**

- `main.py` — entry point; trains a value network and rolls out the resulting controller
- `architectures/` — `pinn.py`, `xtfc.py`, `xtfc_unfreeze.py`
- `problems/` — pendulum, damped pendulum, cart-pole, double integrator, nonlinear system
- `models/`, `visualizers/`, `yamls/` — problem/simulator plumbing, pygame viewer, run configs
- `analytical_scripts/` — MATLAB derivations of the HJB and dynamics

**`RLEnv/` — reinforcement learning** (see `RLEnv/README.md` for detail)

- `invertedpendulum.py`, `baseenv.py` — the shared Gymnasium environment
- `experiment.py` — trains the Stable-Baselines3 algorithms (PPO, A2C, DDPG, TD3, SAC)
- `PIRL/` — the two physics-informed RL algorithms; `main.py` is the entry point
- `comparison/` — trajectory-optimization benchmark and the cross-method plots
- `policy_table/`, `EmbeddedPendulum/` — hardware deployment

## Setup

The two halves have separate environments.

```bash
# PINN / HJB side (repo root)
conda env create -f conda_env.yml     # or: pip install -r requirements.txt

# RL side
conda env create -f RLEnv/environment.yml -n rlenv
```

## Running things

```bash
python main.py                        # solve the HJB PDE with a PINN
python RLEnv/experiment.py            # train the SB3 baselines
python RLEnv/PIRL/main.py             # train PIRL-1 / PIRL-2
```

`RLEnv/comparison/compare.ipynb` reproduces the comparison figures and the table
above.

## Hardware

The learned policy runs on a physical pendulum — a DC motor with an optical
encoder and a 105 g, 13.5 cm arm. Since commanding torque needs motor constants
we don't have, the policy is evaluated offline into a lookup table of
$[\theta_t, \dot\theta_t, \theta_{t+dt}]$ triples and an inner PID position loop
at 1 kHz tracks the commanded angle.
