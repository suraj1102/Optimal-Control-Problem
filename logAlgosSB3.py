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

# ── Raise macOS open-file limit from 256 → 4096 ──────────────────────────────
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, _hard), _hard))

# ── Seeding ──────────────────────────────────────────────────────────────────
seed = 69420
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ── Config ───────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS  = 300_000
N_ENVS           = 32
N_EVAL_ROLLOUTS  = 20        # number of full episodes to average at eval time
SCALE_FACTOR     = 1
YAML_PATH        = "yamls/unfreeze_ip.yaml"
LOG_DIR          = "algo_logs"
MODEL_DIR        = "algo_models"

ALGO_REGISTRY = {
    "SAC":  SAC,
    "TD3":  TD3,
    "DDPG": DDPG,
    "A2C":  A2C,
    "PPO":  PPO,
}

try:
    from sb3_contrib import TQC
    ALGO_REGISTRY["TQC"] = TQC
except ImportError:
    pass


# ── Env factories ─────────────────────────────────────────────────────────────
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
        )
    return _init


# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate_policy(model, problem, scale_factor, n_rollouts: int, logger) -> tuple[float, float]:
    """
    Run `n_rollouts` complete episodes with a deterministic policy.
    Returns (mean_return, std_return) across all rollouts.

    Uses a single (non-vectorised) environment so episode boundaries are
    unambiguous and we don't accidentally mix partial episodes.
    """
    eval_env = make_env_fn(problem, scale_factor)()
    episode_returns = []

    for ep in range(n_rollouts):
        obs, _ = eval_env.reset()
        done = False
        ep_return = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_return += reward
            done = terminated or truncated

        episode_returns.append(ep_return)
        logger.debug(f"    rollout {ep+1}/{n_rollouts}: return = {ep_return:.3f}")

    eval_env.close()
    arr = np.array(episode_returns)
    return float(arr.mean()), float(arr.std())


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    Hyperparams_obj = Hyperparams.from_yaml(YAML_PATH)
    logger = log.get_logger("main")
    logger.setLevel(
        logging.DEBUG if Hyperparams_obj.hyper_params.debug else logging.INFO
    )
    Hyperparams_obj.logger = logger

    problem = damped_inverted_pendulum(Hyperparams_obj)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Phase 1: Train all algorithms ─────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PHASE 1 — Training")
    logger.info("=" * 60)

    for algo_name, AlgoClass in ALGO_REGISTRY.items():
        logger.info(f"\n  [{algo_name}] starting training ...")

        train_env = SubprocVecEnv(
            [make_env_fn(problem, SCALE_FACTOR) for _ in range(N_ENVS)]
        )

        model = AlgoClass("MlpPolicy", train_env, seed=seed, verbose=0)
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            progress_bar=True,
            reset_num_timesteps=True,
        )

        save_path = os.path.join(MODEL_DIR, algo_name)
        model.save(save_path)
        logger.info(f"  [{algo_name}] model saved → {save_path}.zip")

        train_env.close()
        del model, train_env
        gc.collect()
        time.sleep(1)

    # ── Phase 2: Evaluate all algorithms ──────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PHASE 2 — Evaluation")
    logger.info(f"  {N_EVAL_ROLLOUTS} rollouts per algorithm")
    logger.info("=" * 60)

    # Summary CSV: one row per algorithm
    summary_path = os.path.join(LOG_DIR, "eval_summary.csv")
    rollout_path = os.path.join(LOG_DIR, "eval_rollouts.csv")

    with open(summary_path, "w") as sf, open(rollout_path, "w") as rf:
        sf.write("algo,mean_return,std_return,min_return,max_return\n")
        rf.write("algo,rollout,timestep,reward\n")

        for algo_name, AlgoClass in ALGO_REGISTRY.items():
            logger.info(f"\n  [{algo_name}] evaluating ...")

            save_path = os.path.join(MODEL_DIR, algo_name)
            model = AlgoClass.load(save_path)

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

                    ep_rewards.append(reward)

                    # log per timestep
                    rf.write(f"{algo_name},{ep+1},{t},{reward:.6f}\n")

                    t += 1
                    done = terminated or truncated

                ep_return = sum(ep_rewards)

                episode_returns.append(ep_return)
                rf.write(f"{algo_name},{ep+1},{ep_return:.4f}\n")
                rf.flush()
                logger.debug(f"    rollout {ep+1}/{N_EVAL_ROLLOUTS}: {ep_return:.3f}")

            eval_env.close()
            del model

            arr = np.array(episode_returns)
            mean_r = arr.mean()
            std_r  = arr.std()
            min_r  = arr.min()
            max_r  = arr.max()

            sf.write(f"{algo_name},{mean_r:.4f},{std_r:.4f},{min_r:.4f},{max_r:.4f}\n")
            sf.flush()

            logger.info(
                f"  [{algo_name}] mean={mean_r:.3f}  std={std_r:.3f}"
                f"  min={min_r:.3f}  max={max_r:.3f}"
            )

    logger.info(f"\n  Rollout-level log  → {rollout_path}")
    logger.info(f"  Summary            → {summary_path}")
    logger.info("\nDone.")


    
