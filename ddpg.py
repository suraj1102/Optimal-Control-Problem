from models.hparams import Hyperparams
from problems.damped_inverted_pendulum import damped_inverted_pendulum
from environments.pendulum_env import PendulumEnv
from stable_baselines3 import A2C, DDPG, SAC, TD3, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
import log
import logging
import numpy as np
import matplotlib.pyplot as plt
import random

# Seeding
seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Config
TOTAL_TIMESTEPS = 5_000_000
N_ENVS = 32
N_EVAL_ROLLOUTS = 100  # number of full episodes to average at eval time
SCALE_FACTOR = 1
YAML_PATH = "yamls/unfreeze_ip.yaml"
LOG_DIR = "algo_logs"
MODEL_DIR = "algo_models"
LOAD_MODEL = False
MODEL_PATH = "algo_models/ddpg_pendulum.zip"


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

    train_env = SubprocVecEnv(
        [make_env_fn(problem, SCALE_FACTOR) for _ in range(N_ENVS)]
    )

    if LOAD_MODEL:
        logger.info("Loading Model")
        model = DDPG.load(MODEL_PATH, env=train_env, seed=seed)
    else:
        logger.info("Training")
        model = DDPG("MlpPolicy", train_env, seed=seed, verbose=2)
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            progress_bar=True,
            reset_num_timesteps=True,
        )
        model.save(MODEL_PATH)

    train_env.close()

    # Evaluation
    logger.info("Evaluation")
    logger.info(f"{N_EVAL_ROLLOUTS} rollouts per algorithm")

    # Single eval env for clean episode boundaries
    eval_env = make_env_fn(problem, SCALE_FACTOR)()
    episode_returns = []
    cumulative_rewards = []

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

        cumulative = np.cumsum(ep_rewards)
        cumulative_rewards.append(cumulative)

        logger.debug(f" EVAL: rollout {ep + 1}/{N_EVAL_ROLLOUTS}: {ep_return:.3f}")

    eval_env.close()

    ## Plot

    # Find max episode length
    max_len = max(len(cr) for cr in cumulative_rewards)

    # Pad with NaN (so ignored in mean)
    padded = np.full((N_EVAL_ROLLOUTS, max_len), np.nan)

    for i, cr in enumerate(cumulative_rewards):
        padded[i, : len(cr)] = cr

    # Mean and std ignoring NaNs
    mean_curve = np.nanmean(padded, axis=0)
    std_curve = np.nanstd(padded, axis=0)

    # X axis
    timesteps = np.arange(max_len)

    plt.figure()

    plt.plot(timesteps, mean_curve, label="Mean Return")
    plt.fill_between(
        timesteps,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.3,
        label="±1 std",
    )

    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Reward")
    plt.title("Average Cumulative Reward (Evaluation)")
    plt.legend()
    plt.grid()

    plt.show()

    del model


if __name__ == "__main__":
    main()
