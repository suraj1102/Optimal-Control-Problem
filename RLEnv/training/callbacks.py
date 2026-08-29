import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from training.evaluation import run_rollout, plot_rollout


class RewardTrackingCallback(BaseCallback):
    def __init__(self, log_freq: int = 2_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.timesteps: list[int] = []
        self.mean_rewards: list[float] = []
        self._episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])

        if self.num_timesteps % self.log_freq == 0 and self._episode_rewards:
            mean_r = float(np.mean(self._episode_rewards))
            self.timesteps.append(self.num_timesteps)
            self.mean_rewards.append(mean_r)
            self._episode_rewards.clear()

            if self.verbose >= 1:
                print(
                    f"  [train] step={self.num_timesteps:>8,}  mean_reward={mean_r:.4f}"
                )

        return True


class RolloutEvalCallback(BaseCallback):
    def __init__(
        self,
        env_fn,
        run_name: str,
        eval_freq: int = 20_000,
        plot_dir: str = "plots",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.env_fn = env_fn
        self.run_name = run_name
        self.eval_freq = eval_freq
        self.plot_dir = plot_dir

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            thetas, theta_dots, actions, rewards = run_rollout(
                self.model, self.env_fn, deterministic=True
            )
            total_r = sum(rewards)
            title = (
                f"{self.run_name} — step {self.num_timesteps:,}"
                f"  |  episode reward = {total_r:.2f}"
            )
            save_path = os.path.join(
                self.plot_dir,
                self.run_name,
                f"step_{self.num_timesteps:08d}.png",
            )
            plot_rollout(
                thetas, theta_dots, actions, rewards, title=title, save_path=save_path
            )

            if self.verbose >= 1:
                print(
                    f"  [eval] step={self.num_timesteps:>8,}  "
                    f"episode_reward={total_r:.4f}  "
                    f"saved → {save_path}"
                )

        return True
