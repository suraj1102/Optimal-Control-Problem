import yaml
import gymnasium as gym
import PIRL
import utils
import os
import numpy as np
from train import trainAlgo2, Algo1Trainer
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
from training.rewards import make_reward_quadratic #noqa: E402
from training.disturbances import DISTURB_FNS  # noqa: E402

def simulate_env_with_actor(agent, env: gym.Env, max_steps=200):
    """
    Simulate the environment using the agent's actor policy, rendering with 'human' mode.
    Returns trajectory (list of (state, action, reward)) and total reward.
    """
    import matplotlib.pyplot as plt

    plt.ion()
    state, _ = env.reset()
    try:
        env.unwrapped.state = np.array([0.0, 0.0])  # [theta, thetadot] 
        if hasattr(env.unwrapped, 'state') and len(env.unwrapped.state) == 2:
            state = np.array([0.0, 1.0, 0.0])
    except Exception:
        pass
    traj = []
    total_reward = 0.0
    device = next(agent.actor.parameters()).device

    thetas, thetadots, actions = [], [], []
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ylabels = ["theta (rad)", "thetadot (rad/s)", "u (action)"]
    colors = ["b", "g", "r"]
    for ax, ylabel in zip(axs, ylabels):
        ax.set_ylabel(ylabel)
    axs[2].set_xlabel("Step")
    fig.suptitle("Inverted Pendulum Telemetry")

    def extract_theta_thetadot(state):
        if len(state) == 3:
            cos_th, sin_th, thetadot = state
            theta = np.arctan2(sin_th, cos_th)
        elif len(state) == 2:
            theta, thetadot = state
        else:
            theta, thetadot = None, None
        return theta, thetadot

    for t in range(max_steps):
        # Get action from actor
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            action = agent.actor(state_tensor).cpu().numpy()
        if action.ndim > 1:
            action = action.squeeze()

        theta, thetadot = extract_theta_thetadot(state)
        thetas.append(theta)
        thetadots.append(thetadot)
        actions.append(action if np.isscalar(action) else action[0])

        # Live plot update (clear only data, not labels)
        for i, (ax, data, color) in enumerate(
            zip(axs, [thetas, thetadots, actions], colors)
        ):
            ax.clear()
            ax.plot(data, color=color)
            ax.set_ylabel(ylabels[i])
        axs[2].set_xlabel("Step")
        fig.suptitle("Inverted Pendulum Telemetry")
        plt.pause(0.001)

        # Step environment
        state, reward, terminated, truncated, info = env.step(action)
        traj.append((state, action, reward))
        total_reward += reward
        env.render()
        if terminated or truncated:
            break
    plt.ioff()
    plt.show()
    return traj, total_reward


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
    if config["env_name"] == "Pendulum-v1":
        env = gym.make(config["env_name"], render_mode="human")
    elif config["env_name"] == "myPendulum":
        Q = config["cost_matrices"]["Q"]
        R = config["cost_matrices"]["R"]
        reward_fn = make_reward_quadratic(Q[0], Q[1], R[0], normalise=True)
        env = InvertedPendulumEnv(
            reward_fn=reward_fn,
            disturb_fn=DISTURB_FNS["none"],
            dt = 0.05,
            max_steps=200,
            seed=seed,
            gravity=config["system_params"]["g"],
            length=config["system_params"]["l"],
            mass=config["system_params"]["m"]
        )
    manager = utils.Manager(env, config)

    print("ENV created")

    # Instantiate Agent based on Algorithm selection
    if config["algorithm"] == "Algo2":
        agent = PIRL.Algo2(config, env)

        # NOTE: Algo 2 Phase 1: Admissible Policy Initialization
        print("Starting Phase 1: Admissible Policy Search...")
        for i in range(config["init_epochs"]):
            states = manager.get_sobol_samples(config["batch_size"])
            batch = manager.collect_integral_batch(agent, states)
            loss = agent.initialize_policy(batch, P_matrix=np.eye(agent.state_dim))
            if i % 10 == 0:
                print(f"Init Epoch {i} | Loss: {loss:.4f}")

    else:
        # NOTE: Algo 1 requires an explicit dynamic model
        agent = PIRL.Algo1(env, config, env._dynamics_torch_continuous)

    print("agent instantiated")

    # NOTE: Phase 2 & 3: Iterative Policy Iteration
    print(f"Starting {config['algorithm']} Main Training...")
    actor_losses, critic_losses = [], []

    # TODO: Model training
    if config["algorithm"] == "Algo2":
        actor_losses, critic_losses = trainAlgo2(agent)
    else:
        trainer = Algo1Trainer(agent)
        actor_losses, critic_losses = trainer.run()

    # simulate_env_with_actor(agent, env)


if __name__ == "__main__":
    main()
