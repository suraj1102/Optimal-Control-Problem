import yaml
import gymnasium as gym
import PIRL
import utils
import model
import os

def main():
    # 1. Load configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Setup Infrastructure
    env = gym.make(config['env_name'])
    manager = utils.manager(env, config)
    
    # 3. Instantiate Agent based on Algorithm selection
    if config['algorithm'] == "Algo2":
        agent = PIRL.Algo2(config, env)
        
        # --- PHASE 1: Admissible Policy Initialization ---
        print("Starting Phase 1: Admissible Policy Search...")
        for i in range(config['init_epochs']):
            states = manager.get_sobol_samples(config['batch_size'])
            batch = manager.collect_integral_batch(agent, states)
            loss = agent.initialize_policy(batch, P_matrix=np.eye(agent.state_dim))
            if i % 10 == 0: print(f"Init Epoch {i} | Loss: {loss:.4f}")
            
    else:
        # Algo 1 requires an explicit dynamic model
        model = model.PendulumModel()
        agent = PIRL.Algo1(config, env, model)

    # --- PHASE 2 & 3: Iterative Policy Iteration ---
    print(f"Starting {config['algorithm']} Main Training...")
    actor_losses, critic_losses = [], []
    
    for epoch in range(config['total_epochs']):
        # Collect Data
        states = manager.get_sobol_samples(config['batch_size'])
        
        # Training Update
        if config['algorithm'] == "Algo2":
            batch = manager.collect_integral_batch(agent, states)
            loss_c = agent.train_step(batch)
            loss_a = 0 # Algo 2 computes actor loss inside train_step
        else:
            batch = manager.collect_standard_batch(agent, states)
            loss_c = agent.train_step(batch, model.f)
            loss_a = 0
            
        actor_losses.append(loss_a)
        critic_losses.append(loss_c)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Critic Loss: {loss_c:.4f}")
            manager.plot_controller_behavior(agent, epoch)

    # 4. Final Analysis
    manager.plot_convergence(actor_losses, critic_losses)

if __name__ == "__main__":
    main()