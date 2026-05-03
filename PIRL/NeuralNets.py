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


class Actor(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        state_dim, action_dim = get_env_dims(env)
        
        actor_layers = config["model_size"]["actor"]
        self.net = build_sequential(
            state_dim, actor_layers, action_dim, activation=nn.Tanh
        )

    def forward(self, state):
        action = self.net(state)
        return action


class ActorConstraint(Actor):
    def __init__(self, env, config):
        super().__init__(env, config)
        device = torch.device( "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.mps.is_available()
            else "cpu"
        )
        self.umax = torch.tensor(env.action_space.high).to(device=device)
    
    def forward(self, state):
        action = super().forward(state)
        action = torch.tanh(action)
        return action * self.umax
    

class Critic(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        state_dim, _ = get_env_dims(env)
        critic_layers = config["model_size"]["critic"]
        self.net = build_sequential(state_dim, critic_layers, 1, activation=nn.Tanh)

    def forward(self, state):
        value = self.net(state)
        return value


class AdmissibleNet(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        state_dim, action_dim = get_env_dims(env)
        hidden_dim = config["model_size"]["actor"]
        self.net = build_sequential(
            state_dim, [hidden_dim, hidden_dim], action_dim, activation=nn.Tanh
        )

    def forward(self, state):
       action = self.net(state)
       return action


class RunningNormalizer(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(0.0))
        self.eps = eps

    def update(self, x: torch.Tensor):
        # Welford online update
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        n = x.shape[0]
        total = self.count + n
        delta = batch_mean - self.mean
        self.mean += delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta**2 * self.count * n / total) / total
        self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var.sqrt() + self.eps)


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

    
