from architectures.pinn import Pinn
from architectures.xtfc_unfreeze import XTFC_Unfreeze
from architectures.xtfc import XTFC
from models.hparams import Hyperparams
from problems.inverted_pendulum import inverted_pendulum
from problems.damped_inverted_pendulum import damped_inverted_pendulum
from models.env import ProblemEnv
from environments.pendulum_env import PendulumEnv
from models.simulator import Simulator
from stable_baselines3.common.env_checker import check_env

# from visualizers.pendulum import PendulumVisualizer
import torch
import log
import logging
import numpy as np

from stable_baselines3 import A2C

if __name__ == "__main__":
    Hyperparams_obj = Hyperparams.from_yaml("yamls/unfreeze_ip.yaml")
    logger = log.get_logger("main")
    logger.setLevel(logging.INFO if not Hyperparams_obj.hyper_params.debug else logging.DEBUG)
    Hyperparams_obj.logger = logger

    problem = damped_inverted_pendulum(Hyperparams_obj)

    scale_factor = 10

    train_env = PendulumEnv(
        problem, time_step=0.01, max_steps=1000,
        term_radius=0.1, action_bounds=[(-1, 1)], scale_factor=scale_factor
    )
    check_env(train_env, warn=True)

    model = A2C("MlpPolicy", train_env)
    model.learn(total_timesteps=500_000, progress_bar=True)
    train_env.close()

    eval_env = PendulumEnv(
        problem, time_step=0.01, max_steps=1000,
        term_radius=0.1, action_bounds=[(-1, 1)],
        render_mode="human", width=600, height=400,
    )

    obs, _ = eval_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        action *= scale_factor

        obs, reward, terminated, truncated, _ = eval_env.step(action)
        eval_env.render()

        print(f"Reward: {reward:.2f}, Action: {action}")

        if terminated or truncated:
            obs, _ = eval_env.reset()
    eval_env.close()

