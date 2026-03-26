import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


class ProblemEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        problem,
        time_step: float,
        max_steps: int,
        action_bounds: list,
        term_radius: float = None,
    ):
        super().__init__()
        self.problem = problem
        self.hparams = problem.hparams
        self.device = self.hparams.device.device
        self.dt = time_step
        self.max_steps = max_steps
        self.term_radius = term_radius

        ranges = self.hparams.problem_params.input_ranges
        low  = np.array([r[0] for r in ranges], dtype=np.float32)
        high = np.array([r[1] for r in ranges], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        act_low  = np.array([b[0] for b in action_bounds], dtype=np.float32)
        act_high = np.array([b[1] for b in action_bounds], dtype=np.float32)
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        self.Q = self.hparams.problem_params.Q
        self.R = self.hparams.problem_params.R

        self._state: torch.Tensor = None
        self._steps = 0

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _to_numpy(self, t: torch.Tensor) -> np.ndarray:
        return t.squeeze(0).cpu().detach().numpy()

    def _lqr_reward(self, x: torch.Tensor, u: torch.Tensor) -> float:
        x_cost = (x @ self.Q @ x.T).squeeze()
        u_cost = (u @ self.R @ u.T).squeeze()
        return -(x_cost + u_cost).item()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ranges = self.hparams.problem_params.input_ranges
        x0 = np.array(
            [self.np_random.uniform(r[0], r[1]) for r in ranges],
            dtype=np.float32,
        )
        self._state = self._to_tensor(x0)
        self._steps = 0
        return x0, {}

    def step(self, action: np.ndarray):
        u = self._to_tensor(action)
        x = self._state

        with torch.no_grad():
            f_x = self.problem.f_x(x)
            g_x = self.problem.g_x(x)
            x_dot = f_x + g_x * u
            x_next = x + self.dt * x_dot

            prob_name = self.hparams.hyper_params.problem.lower()
            if prob_name == "inverted-pendulum":
                x_next[:, 0] = (x_next[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi

        # reward = self._lqr_reward(x_next, u) - self._steps * 0.5

        theta = x[0, 0]
        omega = x[0, 1]

        reward = torch.cos(theta) + 0.1 * omega**2 + 0.01 * u**2
        reward = -reward
        reward = reward.item()

        self._state = x_next
        self._steps += 1

        obs = self._to_numpy(x_next).astype(np.float32)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

        dist = torch.norm(x_next - self.problem.eq_point).item()
        terminated = bool(self.term_radius is not None and dist < self.term_radius)
        truncated  = bool(self._steps >= self.max_steps)

        return obs, reward, terminated, truncated, {}

    def render(self):
        pass
