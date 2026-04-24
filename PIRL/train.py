import torch
import numpy as np
import matplotlib.pyplot as plt

def train_pirl(agent, manager, config):
    actor_losses = []
    critic_losses = []
    
    # PHASE 1 for Algo 2
    if config['algorithm'] == "Algo2":
        print("Starting Phase 1: Admissible Policy Search...")
        P_init = np.eye(agent.state_dim) * 2.0  # Initial Lyapunov guess
        for _ in range(config['init_epochs']):
            states = manager.get_sobol_samples(config['batch_size'])
            batch = manager.collect_integral_batch(agent, states)
            loss_stab = agent.initialize_policy(batch, P_init)
            
    # PHASE 2 & 3
    print(f"Starting {config['algorithm']} Main Training...")
    for epoch in range(config['total_epochs']):
        # Collect Data
        states = manager.get_sobol_samples(config['batch_size'])
        
        if config['algorithm'] == "Algo2":
            batch = manager.collect_integral_batch(agent, states)
            c_loss, a_loss = agent.update(batch)
        else:
            # Algo 1 Logic (Single step + Dynamics)
            batch = manager.collect_standard_batch(agent, states)
            c_loss, a_loss = agent.update(batch)
            
        actor_losses.append(a_loss)
        critic_losses.append(c_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Critic Loss: {c_loss:.4f} | Actor Loss: {a_loss:.4f}")
            # Optional: Visualize current behavior
            manager.plot_controller_behavior(agent, epoch)

    return actor_losses, critic_losses

def plot_controller_behavior(self, agent, epoch):
        # 1. Simulate one trajectory
        obs, _ = self.env.reset()
        states_history = []
        actions_history = []
        
        for _ in range(200):
            s_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = agent.actor(s_tensor).squeeze(0).numpy()
            obs, _, term, trunc, _ = self.env.step(action)
            states_history.append(obs)
            actions_history.append(action)
            if term or trunc: break
        
        states_history = np.array(states_history)
        
        # 2. Create Plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Controller Behavior at Epoch {epoch}")

        # Panel A: State Trajectories
        axes[0].plot(states_history[:, 0], label='cos(theta)')
        axes[0].plot(states_history[:, 1], label='sin(theta)')
        axes[0].plot(states_history[:, 2], label='angular velocity')
        axes[0].set_title("State Convergence")
        axes[0].legend()

        # Panel B: Control Effort (Torque)
        axes[1].plot(actions_history, color='red')
        axes[1].set_title("Control Signal (u)")
        axes[1].set_ylabel("Torque")

        # Panel C: Value Function Heatmap (Fixed angular velocity = 0)
        # This shows if the PINN has learned the "Physics" of the cost
        res = 50
        x = np.linspace(-1, 1, res)
        y = np.linspace(-1, 1, res)
        X, Y = np.meshgrid(x, y)
        grid_states = torch.tensor(np.stack([X.ravel(), Y.ravel(), np.zeros(res*res)], axis=1), dtype=torch.float32)
        
        with torch.no_grad():
            V = agent.critic(grid_states).reshape(res, res).numpy()
            
        im = axes[2].contourf(X, Y, V, cmap='viridis')
        axes[2].set_title("Critic Value Surface V(x)")
        fig.colorbar(im, ax=axes[2])
        
        plt.tight_layout()
        plt.show()