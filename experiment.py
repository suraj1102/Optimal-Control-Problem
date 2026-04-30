import os
import pickle
import numpy as np
from functools import partial
from copy import deepcopy

from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor

from training.rewards import make_reward_quadratic, make_reward_cos, make_reward_survival
from training.disturbances import DISTURB_FNS
from training.callbacks import RewardTrackingCallback, RolloutEvalCallback
from training.evaluation import plot_training_curves

OUTPUT_DIR = "outputs"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
N_ENVS = 42

for d in (PLOT_DIR, MODEL_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)


def make_env(reward_fn, disturb_fn, env_kwargs: dict):
    from invertedpendulum import InvertedPendulumEnv

    def _fn():
        env = InvertedPendulumEnv(
            reward_fn=reward_fn,
            disturb_fn=disturb_fn,
            **env_kwargs,
        )
        return env

    return _fn


BASE_ENV_KWARGS = dict(
    dt=0.02,
    max_steps=1000,
    action_low=-1.0,
    action_high=1.0,
    init_range=((-0.5, 0.5), (-0.5, 0.5)),
    damping_factor=0.1,
    gravity=10,
    length=0.8,
    mass=0.1,
    failure_termination=(np.pi, 10.0),
    success_termination=None,
)


def train_experiment(exp: dict) -> dict:
    run_name = exp["run_name"]
    model_cls = exp["model_cls"]
    model_kwargs = exp["model_kwargs"]
    reward_fn = exp["reward_fn"]
    disturb_fn = exp["disturb_fn"]
    env_kwargs = exp["env_kwargs"]
    total_steps = exp["total_steps"]
    n_envs = exp["n_envs"]
    log_freq = exp["log_freq"]
    eval_freq = exp["eval_freq"]

    print(f"\n{'=' * 60}")
    print(f"  Starting: {run_name}")
    print(f"  Model:    {model_cls.__name__}")
    print(f"  Steps:    {total_steps:,}  |  n_envs: {n_envs}")
    print(f"{'=' * 60}")

    env_fn = make_env(reward_fn, disturb_fn, env_kwargs)
    vec_env = make_vec_env(env_fn, n_envs=n_envs)

    monitor_path = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(monitor_path, exist_ok=True)

    vec_env = VecMonitor(vec_env, filename=monitor_path)

    model = model_cls(env=vec_env, **model_kwargs)

    reward_cb = RewardTrackingCallback(log_freq=log_freq, verbose=1)
    eval_cb = RolloutEvalCallback(
        env_fn=env_fn,
        run_name=run_name,
        eval_freq=eval_freq,
        plot_dir=PLOT_DIR,
        verbose=1,
    )
    callbacks = CallbackList([reward_cb, eval_cb])

    model.learn(total_timesteps=total_steps, callback=callbacks, progress_bar=True)
    model_path = os.path.join(MODEL_DIR, run_name)
    model.save(model_path)
    print(f"  [save] model → {model_path}.zip")

    thetas, theta_dots, actions, rewards = run_eval_rollout(model, env_fn, run_name)

    result = {
        "timesteps": reward_cb.timesteps,
        "mean_rewards": reward_cb.mean_rewards,
    }

    result_path = os.path.join(RESULTS_DIR, f"{run_name}.pkl")
    with open(result_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  [save] results → {result_path}")

    vec_env.close()
    return result


def run_eval_rollout(model, env_fn, run_name: str):
    from training.evaluation import run_rollout, plot_rollout

    thetas, theta_dots, actions, rewards = run_rollout(
        model, env_fn, deterministic=True
    )
    total_r = sum(rewards)

    plot_rollout(
        thetas,
        theta_dots,
        actions,
        rewards,
        title=f"{run_name} — FINAL  |  episode reward = {total_r:.2f}",
        save_path=os.path.join(PLOT_DIR, run_name, "final_rollout.png"),
    )
    print(f"  [eval] final episode reward = {total_r:.4f}")
    return thetas, theta_dots, actions, rewards


def generate_experiments(
    reward_fns: dict,
    disturb_fns: dict,
    base_env_kwargs: dict,
    total_steps: int = 1_000_000,
):
    MODEL_CONFIGS = {
        # ───────────────────────── PPO ─────────────────────────
        "PPO": dict(
            model_cls=PPO,
            model_kwargs=dict(
                policy="MlpPolicy",
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                verbose=0,
            ),
            n_envs=N_ENVS,
        ),
        # ───────────────────────── A2C ─────────────────────────
        "A2C": dict(
            model_cls=A2C,
            model_kwargs=dict(
                policy="MlpPolicy",
                learning_rate=7e-4,
                n_steps=5,
                gamma=0.99,
                gae_lambda=1.0,
                ent_coef=0.0,
                vf_coef=0.5,
                verbose=0,
            ),
            n_envs=N_ENVS,
        ),
        # ───────────────────────── SAC ─────────────────────────
        "SAC": dict(
            model_cls=SAC,
            model_kwargs=dict(
                policy="MlpPolicy",
                learning_rate=3e-4,
                buffer_size=200_000,
                batch_size=256,
                gamma=0.99,
                tau=0.005,
                train_freq=1,
                gradient_steps=1,
                ent_coef="auto",
                verbose=0,
            ),
            n_envs=N_ENVS,
        ),
        # ───────────────────────── TD3 ─────────────────────────
        "TD3": dict(
            model_cls=TD3,
            model_kwargs=dict(
                policy="MlpPolicy",
                learning_rate=3e-4,  # ↓ fixed (3e-3 is too high)
                buffer_size=200_000,
                batch_size=256,
                gamma=0.99,
                tau=0.005,
                train_freq=1,
                gradient_steps=1,
                policy_delay=2,
                target_policy_noise=0.2,
                target_noise_clip=0.5,
                verbose=0,
            ),
            n_envs=N_ENVS,
        ),
        # ───────────────────────── DDPG ─────────────────────────
        "DDPG": dict(
            model_cls=DDPG,
            model_kwargs=dict(
                policy="MlpPolicy",
                learning_rate=1e-3,
                buffer_size=200_000,
                batch_size=256,
                gamma=0.99,
                tau=0.005,
                train_freq=1,
                gradient_steps=1,
                verbose=0,
            ),
            n_envs=N_ENVS,
        ),
    }

    experiments = []

    for model_name, model_cfg in MODEL_CONFIGS.items():
        for reward_name, reward_fn in reward_fns.items():
            for disturb_name, disturb_fn in disturb_fns.items():
                run_name = f"{model_name}_{reward_name}_{disturb_name}"

                exp = dict(
                    run_name=run_name,
                    model_cls=model_cfg["model_cls"],
                    model_kwargs=deepcopy(model_cfg["model_kwargs"]),
                    reward_fn=reward_fn,
                    disturb_fn=disturb_fn,
                    env_kwargs=deepcopy(base_env_kwargs),
                    total_steps=total_steps,
                    n_envs=model_cfg["n_envs"],
                    log_freq=4_000,
                    eval_freq=50_000,
                )

                experiments.append(exp)

    return experiments


def main(experiments):
    all_results: dict[str, dict] = {}

    for exp in experiments:
        result = train_experiment(exp)
        all_results[exp["run_name"]] = result

    plot_training_curves(
        all_results,
        save_path=os.path.join(PLOT_DIR, "training_curves.png"),
    )
    print("\n[done] all experiments complete.")
    print(f"  plots  → {PLOT_DIR}/")
    print(f"  models → {MODEL_DIR}/")


if __name__ == "__main__":
    rewards = {
        "Q1_1.0_Q2_0.1_R_0.01": make_reward_quadratic(1.0, 0.1, 0.01, normalise=False),
        "Q1_5.0_Q2_0.5_R_0.05": make_reward_quadratic(5.0, 0.5, 0.05, normalise=False),
        "Q1_1.0_Q2_1.0_R_0.1": make_reward_quadratic(1.0, 1.0, 0.1, normalise=False),
        "Q1_1.0_Q2_0.5_R_0.1_norm": make_reward_quadratic(
            1.0, 0.5, 0.1, normalise=True
        ),
        "Q1_3.0_Q2_1.0_R_0.05_norm": make_reward_quadratic(
            3.0, 1.0, 0.05, normalise=True
        ),
        "cos_Q1_1.0_Q2_0.1_R_0.01": make_reward_cos(1.0, 0.1, 0.01),
        "cos_Q1_5.0_Q2_0.5_R_0.05": make_reward_cos(5.0, 0.5, 0.05),
        "cos_Q1_1.0_Q2_1.0_R_0.1": make_reward_cos(1.0, 1.0, 0.1),
        "survival_3": make_reward_survival(deg_threshold=3),
        "survival_6": make_reward_survival(deg_threshold=6),
        "survival_12": make_reward_survival(deg_threshold=12),
    }

    disturbances = {"no_disturbance": DISTURB_FNS["none"]}

    experiments = generate_experiments(
        reward_fns=rewards,
        disturb_fns=disturbances,
        base_env_kwargs=BASE_ENV_KWARGS,
        total_steps=1_000_000,
    )

    main(experiments=experiments)
