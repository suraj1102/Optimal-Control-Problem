import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv


def run_rollout(model, env_fn, deterministic: bool = True):
    env = env_fn()
    obs, _ = env.reset()

    thetas, theta_dots, actions, rewards = [], [], [], []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        thetas.append(float(obs[0]))
        theta_dots.append(float(obs[1]))
        actions.append(float(action[0]))
        rewards.append(float(reward))

    env.close()
    return thetas, theta_dots, actions, rewards


def plot_rollout(
    thetas,
    theta_dots,
    actions,
    rewards,
    title: str = "Rollout",
    save_path: str | None = None,
):
    t = np.arange(len(thetas))
    cum_reward = np.cumsum(rewards)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, thetas, color="steelblue")
    ax1.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax1.set_title("Angle θ (rad)")
    ax1.set_xlabel("Step")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, theta_dots, color="darkorange")
    ax2.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax2.set_title("Angular velocity θ̇ (rad/s)")
    ax2.set_xlabel("Step")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.step(t, actions, color="forestgreen", where="mid")
    ax3.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax3.set_title("Control input u")
    ax3.set_xlabel("Step")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, cum_reward, color="crimson")
    ax4.set_title(f"Cumulative reward  (total={cum_reward[-1]:.2f})")
    ax4.set_xlabel("Step")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  [plot] saved → {save_path}")

    plt.close(fig)
    return fig


def plot_training_curves(
    results: dict[str, dict],
    save_path: str | None = None,
):
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, data in results.items():
        xs = data["timesteps"]
        ys = data["mean_rewards"]
        ax.plot(xs, ys, label=name, linewidth=1.5)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Training curves")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  [plot] saved → {save_path}")

    plt.close(fig)
    return fig
