from models.hparams import Hyperparams
from problems.damped_inverted_pendulum import damped_inverted_pendulum
from environments.pendulum_env import PendulumEnv
from stable_baselines3 import A2C, DDPG, SAC, TD3, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import torch
import log
import logging
import numpy as np
import random
import multiprocessing as mp
import os
import gc
import time
import resource

# Seeding
seed = 69420
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Config
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 32
N_EVAL_ROLLOUTS = 20  # number of full episodes to average at eval time
SCALE_FACTOR = 1
YAML_PATH = "yamls/unfreeze_ip.yaml"
LOG_DIR = "algo_logs"
MODEL_DIR = "algo_models"


def make_env_fn(problem, scale_factor):
    """Returns a zero-arg callable that creates one PendulumEnv."""

    def _init():
        return PendulumEnv(
            problem,
            time_step=0.01,
            max_steps=1000,
            term_radius=0.1,
            action_bounds=[(-1, 1)],
            scale_factor=scale_factor,
            render_mode="human",
        )

    return _init


def main():
    Hyperparams_obj = Hyperparams.from_yaml(YAML_PATH)
    logger = log.get_logger("main")
    logger.setLevel(
        logging.DEBUG if Hyperparams_obj.hyper_params.debug else logging.INFO
    )
    Hyperparams_obj.logger = logger

    problem = damped_inverted_pendulum(Hyperparams_obj)

    # Train
    logger.info("Training")

    train_env = SubprocVecEnv(
        [make_env_fn(problem, SCALE_FACTOR) for _ in range(N_ENVS)]
    )

    model = DDPG("MlpPolicy", train_env, seed=seed, verbose=2)
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
        reset_num_timesteps=True,
    )

    train_env.close()

    # Evaluation
    logger.info("Evaluation")
    logger.info(f"{N_EVAL_ROLLOUTS} rollouts per algorithm")

    # Single eval env for clean episode boundaries
    eval_env = make_env_fn(problem, SCALE_FACTOR)()
    episode_returns = []

    for ep in range(N_EVAL_ROLLOUTS):
        obs, _ = eval_env.reset()
        done = False
        t = 0
        ep_rewards = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            eval_env.render()  # Render the environment at each step
            ep_rewards.append(reward)
            t += 1
            done = terminated or truncated

        ep_return = sum(ep_rewards)

        episode_returns.append(ep_return)
        logger.debug(f" EVAL: rollout {ep + 1}/{N_EVAL_ROLLOUTS}: {ep_return:.3f}")

    eval_env.close()
    del model

    arr = np.array(episode_returns)
    mean_r = arr.mean()
    std_r = arr.std()
    min_r = arr.min()
    max_r = arr.max()

    logger.info(f"{mean_r=}")
    logger.info(f"{std_r=}")
    logger.info(f"{min_r=}")
    logger.info(f"{max_r=}")


def gymenv_only():
    import gymnasium as gym
    import numpy as np
    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import NormalActionNoise

    # ---- Config ----
    ENV_ID = "Pendulum-v1"
    TRAIN_STEPS = 1_000_000
    EVAL_EPISODES_PER_ENV = 1
    NUM_EVAL_ENVS = 20
    SEED = 69420

    # ---- Create training env ----
    def make_env_fn():
        def _init():
            return gym.make(ENV_ID)

        return _init

    train_env = SubprocVecEnv([make_env_fn() for _ in range(N_ENVS)])

    train_env.reset()

    # ---- Action noise (DDPG needs exploration noise) ----
    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )

    # ---- Model ----
    model = DDPG(
        "MlpPolicy",
        train_env,
        # action_noise=action_noise,
        verbose=2,
        seed=SEED,
    )

    # ---- Train ----
    model.learn(total_timesteps=TRAIN_STEPS, progress_bar=True)

    # ---- Save (optional) ----
    model.save("./algo_models/ddpg_pendulum")

    # ---- Evaluation loop: 20 envs one by one ----
    for i in range(NUM_EVAL_ENVS):
        env = gym.make(ENV_ID, render_mode="human")
        obs, _ = env.reset(seed=SEED + i)

        done = False
        truncated = False
        ep_reward = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward

        print(f"Env {i + 1}: episode reward = {ep_reward:.2f}")
        env.close()

    train_env.close()


if __name__ == "__main__":
    main()
