import yaml
import gymnasium as gym
import PIRL
import os
import numpy as np
from train import Algo1Trainer
import torch
import sys

# Ensure imports work when running this file from within the "PIRL" directory.
_here = os.path.dirname(os.path.abspath(__file__))
_cwd = os.getcwd()

if os.path.basename(_cwd).lower() == "pirl":
    # running as: (repo)/PIRL$ python main.py  -> add repo root (parent of PIRL)
    repo_root = os.path.abspath(os.path.join(_cwd, os.pardir))
else:
    # running from inside package tree -> add parent of this file (repo root candidate)
    repo_root = os.path.abspath(os.path.join(_here, os.pardir))

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from invertedpendulum import InvertedPendulumEnv  # noqa: E402
from training.rewards import make_reward_quadratic  # noqa: E402
from training.disturbances import DISTURB_FNS  # noqa: E402


def simulate(agent: PIRL.PIRL, t: float = 10):
    env = agent.env
    device = next(agent.actor.parameters()).device
    agent.actor.eval()

    x0 = torch.tensor([[0.1, 0.0]], device=device)
    dt = env.dt
    numPoints = int(t / dt)

    x_list = []
    x_list.append(x0)

    for _ in range(numPoints):
        # Get action from actor
        with torch.no_grad():
            ut = agent.actor(x0)
        ut_np = ut.detach().cpu().numpy().flatten()
        # Euler integration for continuous dynamics: x_{t+1} = x_t + F(x_t, u_t) * dt
        with torch.no_grad():
            delta = (
                agent.F(
                    x0,
                    torch.tensor(
                        ut_np.reshape(1, -1), device=device, dtype=torch.float32
                    ),
                )
                * dt
            )
        x1 = x0 + delta
        # Wrap theta to [-pi, pi]
        x1_np = x1.detach().cpu().numpy()
        x1_np[0, 0] = (x1_np[0, 0] + np.pi) % (2 * np.pi) - np.pi
        x0 = torch.tensor(x1_np, device=device, dtype=torch.float32)
        x_list.append(x0)

    # Convert to numpy for plotting
    x_arr = torch.stack(x_list).cpu().numpy()[:, 0, :]
    t_arr = np.arange(x_arr.shape[0]) * dt

    return x_arr, t_arr


def phase_plot(agent: PIRL.PIRL):
    import matplotlib.pyplot as plt

    radii_deg = [30, 50, 90, 120, 150, 180]  # List of radii in degrees
    num_traj = 6
    t = 10
    trajs = []
    for radius_deg in radii_deg:
        radius = np.deg2rad(radius_deg)
        for i in range(num_traj):
            angle = 2 * np.pi * i / num_traj
            theta0 = radius * np.cos(angle)
            theta_dot0 = radius * np.sin(angle)
            agent.actor.eval()
            x0 = torch.tensor(
                [[theta0, theta_dot0]],
                dtype=torch.float32,
                device=next(agent.actor.parameters()).device,
            )
            # Use simulate with custom x0
            env = agent.env
            dt = env.dt
            numPoints = int(t / dt)
            x_list = [x0]
            x = x0
            for _ in range(numPoints):
                with torch.no_grad():
                    ut = agent.actor(x)
                ut_np = ut.detach().cpu().numpy().flatten()
                with torch.no_grad():
                    delta = (
                        agent.F(
                            x,
                            torch.tensor(
                                ut_np.reshape(1, -1),
                                device=x.device,
                                dtype=torch.float32,
                            ),
                        )
                        * dt
                    )
                x1 = x + delta
                x1_np = x1.detach().cpu().numpy()
                x1_np[0, 0] = (x1_np[0, 0] + np.pi) % (2 * np.pi) - np.pi
                x = torch.tensor(x1_np, device=x.device, dtype=torch.float32)
                x_list.append(x)
            x_arr = torch.stack(x_list).cpu().numpy()[:, 0, :]
            trajs.append(x_arr)

    # Plot all trajectories on phase plot
    plt.figure(figsize=(7, 5))
    for arr in trajs:
        plt.plot(arr[:, 0], arr[:, 1], lw=2)
        plt.scatter(arr[0, 0], arr[0, 1], color="b", marker="o")  # Start point
        plt.scatter(
            arr[-1, 0], arr[-1, 1], color="k", marker="o", s=15, zorder=5
        )  # End point (smaller dot)
    plt.xlabel("theta (rad)")
    plt.ylabel("theta_dot (rad/s)")
    plt.title("Phase plot of multiple rollouts")
    plt.xlim(-np.pi, np.pi)
    plt.grid()
    plt.tight_layout()
    plt.show()


def main():
    # Set all seeds for reproducibility
    import random

    seed = 7
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    gym.utils.seeding.np_random(seed)

    # Set torch device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
    print(f"Using torch device: {device}")

    # Load config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("Config Loaded")

    # Setup env
    Q = config["cost_matrices"]["Q"]
    R = config["cost_matrices"]["R"]
    reward_fn = make_reward_quadratic(Q[0], Q[1], R[0], normalise=True)
    env = InvertedPendulumEnv(
        reward_fn=reward_fn,
        disturb_fn=DISTURB_FNS["none"],
        dt=0.02,
        max_steps=200,
        seed=seed,
        gravity=config["system_params"]["g"],
        length=config["system_params"]["l"],
        mass=config["system_params"]["m"],
        damping_factor=config["system_params"]["b"],
        theta_dot_limit=(-4, 4),
    )

    print("ENV created")

    # Select Agend Based On Algorithm
    if config["algorithm"] == "Algo2":
        raise NotImplementedError()
    else:
        # Algo 1 requires an explicit dynamic model
        agent = PIRL.Algo1(env, config, env._dynamics_torch_continuous)

    print("agent instantiated")

    print(f"Starting {config['algorithm']} Main Training...")
    actor_losses, critic_losses = [], []

    if config["algorithm"] == "Algo2":
        raise NotImplementedError()
    else:
        trainer = Algo1Trainer(agent)
        actor_losses, critic_losses = trainer.run()

    # simulate(agent)
    phase_plot(agent)


if __name__ == "__main__":
    main()
