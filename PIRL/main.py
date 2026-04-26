import yaml
import gymnasium as gym
import PIRL
import utils
import SystemModels
import os
import numpy as np
from train import trainAlgo2, Algo1Trainer



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
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Using torch device: {device}")
    

    # Load config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("Config Loaded")

    # Setup env
    env = gym.make(config["env_name"])
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
        trainer.run()
        
if __name__ == "__main__":
    main()
