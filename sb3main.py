from models.hparams import Hyperparams
from problems.damped_inverted_pendulum import damped_inverted_pendulum
from environments.pendulum_env import PendulumEnv
from stable_baselines3 import A2C, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
import log
import logging
import numpy as np
import random
import multiprocessing as mp

seed = 69420

random.seed(seed)
np.random.seed(seed)       # NumPy
torch.manual_seed(seed)    # PyTorch (CPU)

def make_env(rank, scale_factor):
    def _init():
        env = PendulumEnv(
            problem, time_step=0.01, max_steps=1000,
            term_radius=0.1, action_bounds=[(-1, 1)], scale_factor=scale_factor
        )
        return env
    return _init

N_ENVS = 32 # or os.cpu_count()

if __name__ == "__main__":
    mp.set_start_method("fork", force=True) # MacOS needs this
    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    Hyperparams_obj = Hyperparams.from_yaml("yamls/unfreeze_ip.yaml")
    logger = log.get_logger("main")
    logger.setLevel(logging.INFO if not Hyperparams_obj.hyper_params.debug else logging.DEBUG)
    Hyperparams_obj.logger = logger

    problem = damped_inverted_pendulum(Hyperparams_obj)

    scale_factor = 1

    train_env = SubprocVecEnv([make_env(i, scale_factor) for i in range(N_ENVS)])

    model = DDPG("MlpPolicy", train_env, seed=seed)
    model.learn(total_timesteps=100_000, progress_bar=True)
    train_env.close()

    eval_env = PendulumEnv(
        problem, time_step=0.01, max_steps=1000,
        term_radius=0.1, action_bounds=[(-1, 1)],
        render_mode="human", width=600, height=400, scale_factor=scale_factor
    )

    obs, _ = eval_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = eval_env.step(action)
        eval_env.render()

        print(f"Reward: {reward:.2f}, Action: {action}")

        if terminated or truncated:
            obs, _ = eval_env.reset()
            print(terminated, truncated)
    eval_env.close()

