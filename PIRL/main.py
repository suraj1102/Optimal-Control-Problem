import yaml
import gymnasium as gym
import PIRL
import utils
import SystemModels
import os
import numpy as np
from train import trainAlgo2, Algo1Trainer
import torch

def plot_value_function(agent, env):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # Value function is the critic
    critic = agent.critic
    device = next(critic.parameters()).device

    # Value function is defined on (cos(theta), sin(theta), thetadot)
    # We'll sweep theta in [-pi, pi], thetadot in env.observation_space.high[-1]
    n_theta = 100
    n_thetadot = 100
    theta_range = np.linspace(-np.pi, np.pi, n_theta)
    # Use env.observation_space for thetadot limits
    thetadot_low = env.observation_space.low[-1]
    thetadot_high = env.observation_space.high[-1]
    thetadot_range = np.linspace(thetadot_low, thetadot_high, n_thetadot)

    Theta, Thetadot = np.meshgrid(theta_range, thetadot_range)
    # Prepare state grid: (cos(theta), sin(theta), thetadot)
    cos_theta = np.cos(Theta)
    sin_theta = np.sin(Theta)
    state_grid = np.stack([cos_theta, sin_theta, Thetadot], axis=-1)
    state_grid_flat = state_grid.reshape(-1, 3)

    # Evaluate value function
    with torch.no_grad():
        state_tensor = torch.tensor(state_grid_flat, dtype=torch.float32, device=device)
        values = critic(state_tensor).cpu().numpy().reshape(n_thetadot, n_theta)

    # Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Theta, Thetadot, values, cmap='viridis')
    ax.set_xlabel('theta (rad)')
    ax.set_ylabel('thetadot (rad/s)')
    ax.set_zlabel('Value')
    ax.set_title('Value Function Surface')
    plt.show()

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
    print("HELLO--")

    # Set all seeds for reproducibility
    import random
    import torch

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
    env = gym.make(config["env_name"], render_mode="human")
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
        system_model = SystemModels.PendulumModel()
        agent = PIRL.Algo1(env, config, system_model)

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

    plot_value_function(agent, env)
    simulate_env_with_actor(agent, env)


if __name__ == "__main__":
    main()
