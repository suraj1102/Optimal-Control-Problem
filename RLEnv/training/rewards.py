import numpy as np
from typing import Callable

"""
LQR loss (experiment with different L, Q, and R)
Cos theta wala loss
Some time dependent loss

"""


def make_reward_quadratic(
    Q1: float, Q2: float, R: float, normalise: bool = False
) -> Callable:
    if not normalise:

        def _fn(state: np.ndarray, action: np.ndarray) -> float:
            theta, theta_dot = state
            u = action[0]
            return -(Q1 * theta**2 + Q2 * theta_dot**2 + R * u**2)

    else:

        def _fn(state: np.ndarray, action: np.ndarray) -> float:
            theta, theta_dot = state
            theta_dot = np.clip(theta_dot, -8 * np.pi, 8 * np.pi)
            theta_dot /= 8 * np.pi
            theta /= np.pi
            u = action[0]

            return -(Q1 * theta**2 + Q2 * theta_dot**2 + R * u**2)

    return _fn

def make_reward_cos(
        Q1: float, Q2: float, R: float
) -> Callable:
    def _fn(state: np.ndarray, action: np.ndarray) -> float:
        theta, theta_dot = state
        u = action[0]

        return Q1 * np.cos(theta) - Q2 * theta_dot - R * u
    
    return _fn

def make_reward_survival(deg_threshold: float = 6) -> Callable:
    def _fn(state: np.ndarray, action: np.ndarray) -> float:
        theta, theta_dot = state
        u = action[0]
        
        threshold = np.deg2rad(deg_threshold)
        condition =  theta < threshold and theta > -threshold
        reward = 1 if condition else -0.1
        return reward
    
    return _fn


REWARD_FNS = {
    "quadratic": make_reward_quadratic,
    "cosine": make_reward_cos,
    "survival": make_reward_survival
    }
