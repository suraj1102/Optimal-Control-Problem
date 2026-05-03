from scipy.stats import qmc
import matplotlib.pyplot as plt
import numpy as np
import torch
import gymnasium as gym

def sample_states(env: gym.Env, n_samples: int):
    """Sample Sobol points from the state space of the given environment."""
    state_dim = env.observation_space.shape[0]
    sampler = qmc.Sobol(d=state_dim, scramble=True)
    raw_samples = sampler.random(n=n_samples)
    low = env.observation_space.low
    high = env.observation_space.high

    low = np.asarray(low, dtype=np.float32).copy()
    high = np.asarray(high, dtype=np.float32).copy()
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        print("[SAMPLE_STATES: WARNING] State Space as Inf Limits")
        low = np.array([-np.pi, -4.0], dtype=np.float32)
        high = np.array([np.pi, 4.0], dtype=np.float32)

    samples = low + raw_samples * (high - low)
    return torch.tensor(samples, dtype=torch.float32)
