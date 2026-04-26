# imports
import torch
import torch.nn as nn


def get_env_dims(env):
    state_dim = env.observation_space.shape[0]
    if hasattr(env.action_space, "shape"):
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    return state_dim, action_dim


def get_env_bounds(env):
    state_low = env.observation_space.low
    state_high = env.observation_space.high
    action_low = env.action_space.low
    action_high = env.action_space.high
    return state_low, state_high, action_low, action_high


def build_sequential(input_dim, layer_sizes, output_dim=None, activation=nn.Tanh):
    layers = []
    prev_dim = input_dim
    for size in layer_sizes:
        layers.append(nn.Linear(prev_dim, size))
        layers.append(activation())
        prev_dim = size
    if output_dim is not None:
        layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class StateActionNormalizer:
    def __init__(self, state_low, state_high, action_low, action_high):
        self.state_low = torch.tensor(state_low, dtype=torch.float32)
        self.state_high = torch.tensor(state_high, dtype=torch.float32)
        self.action_low = torch.tensor(action_low, dtype=torch.float32)
        self.action_high = torch.tensor(action_high, dtype=torch.float32)

    def normalize_state(self, state):
        state_low = self.state_low.to(state.device)
        state_high = self.state_high.to(state.device)
        return 2 * (state - state_low) / (state_high - state_low) - 1

    def denormalize_state(self, norm_state):
        state_low = self.state_low.to(norm_state.device)
        state_high = self.state_high.to(norm_state.device)
        return (norm_state + 1) * (state_high - state_low) / 2 + state_low

    def normalize_action(self, action):
        action_low = self.action_low.to(action.device)
        action_high = self.action_high.to(action.device)
        return 2 * (action - action_low) / (action_high - action_low) - 1

    def denormalize_action(self, norm_action):
        action_low = self.action_low.to(norm_action.device)
        action_high = self.action_high.to(norm_action.device)
        return (norm_action + 1) * (action_high - action_low) / 2 + action_low


class Actor(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        state_dim, action_dim = get_env_dims(env)
        state_low, state_high, action_low, action_high = get_env_bounds(env)
        self.normalizer = StateActionNormalizer(
            state_low, state_high, action_low, action_high
        )
        actor_layers = config["model_size"]["actor"]
        self.net = build_sequential(
            state_dim, actor_layers, action_dim, activation=nn.Tanh
        )
        self.action_low = torch.tensor(action_low, dtype=torch.float32)
        self.action_high = torch.tensor(action_high, dtype=torch.float32)

    def forward(self, state):
        # Normalize state
        norm_state = self.normalizer.normalize_state(state)
        action = self.net(norm_state)
        # Output in [-1, 1], scale to env action bounds
        scaled_action = self.normalizer.denormalize_action(action)
        return scaled_action


class Critic(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        state_dim, _ = get_env_dims(env)
        critic_layers = config["model_size"]["critic"]
        self.net = build_sequential(state_dim, critic_layers, 1, activation=nn.Tanh)

    def forward(self, state):
        return self.net(state)


# FIXME: Use same network as the actor/critic [check in paper] 
class AdmissibleNet(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        state_dim, action_dim = get_env_dims(env)
        state_low, state_high, action_low, action_high = get_env_bounds(env)
        self.normalizer = StateActionNormalizer(
            state_low, state_high, action_low, action_high
        )
        hidden_dim = config["hyperparameters"]["hidden_dim"]
        self.net = build_sequential(
            state_dim, [hidden_dim, hidden_dim], action_dim, activation=nn.Tanh
        )
        self.action_low = torch.tensor(action_low, dtype=torch.float32)
        self.action_high = torch.tensor(action_high, dtype=torch.float32)

    def forward(self, state):
        norm_state = self.normalizer.normalize_state(state)
        action = self.net(norm_state)
        scaled_action = self.normalizer.denormalize_action(action)
        return scaled_action


if __name__ == "__main__":
    import os
    import gymnasium as gym
    import yaml
    import utils

    # Load config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("Config Loaded")

    # Setup env
    env = gym.make(config["env_name"])
    manager = utils.Manager(env, config)

    print("Env created")


    # Instantiate networks
    actorNN = Actor(env, config)
    criticNN = Critic(env, config)
    admissibleNN = AdmissibleNet(env, config)

    print("\nActor network:")
    print(actorNN.net)
    print("\nCritic network:")
    print(criticNN.net)
    print("\nAdmissibleNet network:")
    print(admissibleNN.net)
    print("\nActivations used: nn.Tanh")

    # Test with a random state from the environment
    state, _ = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32)

    # Actor output
    actor_action = actorNN(state_tensor)
    print("\nActor output (action):", actor_action.detach().numpy())

    # Critic output
    critic_value = criticNN(state_tensor)
    print("Critic output (value):", critic_value.detach().numpy())

    # AdmissibleNet output
    admissible_action = admissibleNN(state_tensor)
    print("AdmissibleNet output (action):", admissible_action.detach().numpy())

    
