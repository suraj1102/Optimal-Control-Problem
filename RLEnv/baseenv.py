import numpy as np
import gymnasium as gym
from typing import Optional, Any
from numpy.random import Generator


class BaseEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        dt: float = 0.02,
        max_steps: int = 500,
        action_low: float = -1.0,
        action_high: float = 1.0,
        states_low: list[float] = [-np.pi, -8],
        states_high: list[float] = [np.pi, 8],
        state_dim: int = 2,
        action_dim: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.dt = dt
        self.max_steps = max_steps
        self.step_count = 0

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.action_space = gym.spaces.Box(
            low=action_low, high=action_high, shape=(action_dim,), dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=np.array(states_low, dtype=np.float32),
            high=np.array(states_high, dtype=np.float32),
            shape=(state_dim,), dtype=np.float32
        )

        self.state: np.ndarray
        self.np_random: Generator

        self.reset(seed=seed)

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.state = self._initial_state()
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.state = self._dynamics(self.state, action)
        self.state = self._disturb_state(self.state)
        self.step_count += 1

        obs = self._get_obs()
        reward = self._reward(self.state, action)
        terminated = self._is_terminated(self.state)
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}

    # VIRTUAL FUNCTIONS

    def _initial_state(self) -> np.ndarray:
        raise NotImplementedError

    def _dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _reward(self, state: np.ndarray, action: np.ndarray) -> float:
        raise NotImplementedError

    def _is_terminated(self, state: np.ndarray) -> bool:
        return False

    def _get_obs(self) -> np.ndarray:
        return self.state.astype(np.float32)

    def _disturb_state(self, state: np.ndarray) -> np.ndarray:
        return state
