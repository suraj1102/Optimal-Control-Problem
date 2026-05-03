from baseenv import BaseEnv
import numpy as np
from typing import Optional, Callable
import torch


class InvertedPendulumEnv(BaseEnv):
    def __init__(
        self,
        reward_fn: Callable[[np.ndarray, np.ndarray], float],
        disturb_fn: Callable[[np.ndarray], np.ndarray],
        dt: float = 0.02,
        max_steps: int = 500,
        action_low: float = -1,
        action_high: float = 1,
        state_dim: int = 2,
        action_dim: int = 1,
        seed: Optional[int] = None,
        init_range: tuple = ((-0.5, 0.5), (-0.5, 0.5)),
        damping_factor: float = 0,
        gravity: float = 9.8,
        length: float = 1,
        mass: float = 1,
        success_termination: Optional[tuple] = None,
        failure_termination: Optional[tuple] = None,
        theta_dot_limit: Optional[tuple] = (-8, 8),
    ):
        self.damping_factor = damping_factor
        self.gravity = gravity
        self.length = length
        self.mass = mass
        self.reward_fn = reward_fn
        self.disturb_fn = disturb_fn
        self.init_range = init_range
        self.success_termination = success_termination
        self.failure_termination = failure_termination

        states_low = [-np.pi, theta_dot_limit(0)]
        states_high = [np.pi, theta_dot_limit(1)]

        super().__init__(
            dt,
            max_steps,
            action_low,
            action_high,
            states_low,
            states_high,
            state_dim,
            action_dim,
            seed,
        )

    def _initial_state(self) -> np.ndarray:
        low = np.array([b[0] for b in self.init_range])
        high = np.array([b[1] for b in self.init_range])

        return self.np_random.uniform(low=low, high=high)

    def _dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        theta = state[0]
        theta_dot = state[1]
        u = action[0]

        theta_ddot = (
            self.gravity / self.length * np.sin(theta)
            - (self.damping_factor / self.mass) * theta_dot
            + u / (self.mass * self.length**2)
        )

        new_theta_dot = theta_dot + self.dt * theta_ddot
        new_theta = theta + self.dt * new_theta_dot

        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi  # wrap to -pi to pi
        return np.array([new_theta, new_theta_dot], dtype=np.float32)

    def _dynamics_torch(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        theta = state[:, 0]
        theta_dot = state[:, 1]
        u = action[:, 0]  # shape: [batch]

        theta_ddot = (
            self.gravity / self.length * torch.sin(theta)
            - (self.damping_factor / self.mass) * theta_dot
            + u / (self.mass * self.length**2)
        )

        # Euler Integration
        new_theta_dot = theta_dot + self.dt * theta_ddot
        new_theta = theta + self.dt * new_theta_dot

        new_theta = (new_theta + torch.pi) % (
            2 * torch.pi
        ) - torch.pi  # wrap to -pi to pi

        # Stack along dim=1 to get shape [batch, 2]
        return torch.stack([new_theta, new_theta_dot], dim=1)

    def _dynamics_torch_continuous(self, state, action):
        """Used to solve continuous time ricatti equation for lqr initialization"""
        theta = state[:, 0]
        theta_dot = state[:, 1]
        u = action[:, 0]

        theta_ddot = (
            self.gravity / self.length * torch.sin(theta)
            - (self.damping_factor / self.mass) * theta_dot
            + u / (self.mass * self.length**2)
        )

        return torch.stack([theta_dot, theta_ddot], dim=1)

    def _reward(self, state: np.ndarray, action: np.ndarray) -> float:
        return self.reward_fn(state, action)

    def _is_terminated(self, state: np.ndarray) -> bool:
        theta, theta_dot = state

        if self.failure_termination is not None:
            theta_bound, theta_dot_bound = self.failure_termination
            if abs(theta) > theta_bound or abs(theta_dot) > theta_dot_bound:
                return True

        if self.success_termination is not None:
            eps_theta, eps_theta_dot = self.success_termination
            if abs(theta) < eps_theta and abs(theta_dot) < eps_theta_dot:
                return True

        return False

    def _get_obs(self) -> np.ndarray:
        return super()._get_obs()

    def _disturb_state(self, state: np.ndarray) -> np.ndarray:
        return self.disturb_fn(state)
