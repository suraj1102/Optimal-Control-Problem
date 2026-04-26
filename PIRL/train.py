import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils import Manager, sample_states
from PIRL import Algo1, Algo2


def init_weights(m):
    """
    Initialize Weights for NN
    Usage: nnObject.apply(init_weights)
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class Algo1Trainer:
    def __init__(self, agent: Algo1):
        self.F = agent.F
        self.env = agent.env
        self.config = agent.config
        self.actorNN: nn.Module = agent.actor
        self.criticNN: nn.Module = agent.critic
        self.Q = agent.Q
        self.R = agent.R
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.mps.is_available()
            else "cpu"
        )
        self.kmax = int(self.config.get("kmax", 10))
        self.epsilon_v = float(self.config.get("epsilon_v", 10))
        self.epsilon_u = float(self.config.get("epsilon_u", 1e-2))
        self.Nepochs = int(self.config.get("Nepochs", 100))
        self.batch_size = int(self.config.get("batch_size", 2**3))
        self.print_every = int(self.config.get("print_every", 200))
        self.opt_actor = agent.opt_actor
        self.opt_critic = agent.opt_critic
        self.k = 0
        self.actor_losses, self.critic_losses = [], []
        self.L_v = torch.tensor(float("inf"), device=self.device)
        self.L_u_prev = torch.tensor(float("inf"), device=self.device)

        # Send tensors to device
        self.actorNN.to(self.device)
        self.criticNN.to(self.device)
        self.Q = agent.Q.to(self.device)
        self.R = agent.R.to(self.device)

    def initialize_critic(self):
        with torch.no_grad():
            n_init = 2048
            x_init = sample_states(self.env, n_init).to(self.device)
            y_init = torch.einsum("bi,ij,bj->b", x_init, self.Q, x_init).unsqueeze(1)
        self.criticNN.train()
        init_optimizer = torch.optim.Adam(self.criticNN.parameters(), lr=1e-2)
        for init_iter in range(int(0.20 * self.Nepochs)):
            idx = torch.randint(0, n_init, (256,))
            x_batch = x_init[idx]
            y_batch = y_init[idx]
            pred = self.criticNN(x_batch)
            loss = torch.nn.functional.mse_loss(pred, y_batch)
            init_optimizer.zero_grad()
            loss.backward()
            init_optimizer.step()
            if init_iter % self.print_every == 0:
                print(f"[INIT CRITIC]: iter={init_iter}, fit xTQx loss={loss:.6f}")

    def initialize_actor(self):
        self.actorNN.apply(init_weights)
        for epoch_iter in range(int(0.2 * self.Nepochs)):
            states = sample_states(self.env, self.batch_size).to(self.device)
            states.requires_grad_(True)
            actions = self.actorNN(states)
            Fxu_np = self.F(
                states.cpu().detach().numpy(), actions.cpu().detach().numpy()
            )
            Fxu = torch.tensor(Fxu_np, dtype=torch.float32, device=self.device)
            V = self.criticNN(states).squeeze()
            grad_V = torch.autograd.grad(V.sum(), states, create_graph=True)[0]
            xQx = torch.einsum("bi,ij,bj->b", states, self.Q, states)
            uRu = torch.einsum("bi,ij,bj->b", actions, self.R, actions)
            lu_terms = xQx + uRu + (grad_V * Fxu).sum(dim=1)
            L_u = lu_terms.mean()
            self.opt_actor.zero_grad()
            L_u.backward()
            self.opt_actor.step()
            if epoch_iter % self.print_every == 0:
                print(
                    f"[Phase 0] iter={epoch_iter} | L_u={L_u.item():.6f} | L_u_prev={self.L_u_prev.item():.6f}"
                )
        self.L_u_prev = L_u

    def policy_evaluation(self):
        self.L_v = torch.tensor(float("inf"), device=self.device)
        while self.L_v.item() > self.epsilon_v:
            self.criticNN.apply(init_weights)
            for epoch_iter in range(self.Nepochs):
                states = sample_states(self.env, self.batch_size).to(self.device)
                states.requires_grad_(True)
                with torch.no_grad():
                    actions = self.actorNN(states)
                Fxu_np = self.F(
                    states.cpu().detach().numpy(), actions.cpu().detach().numpy()
                )
                Fxu = torch.tensor(Fxu_np, dtype=torch.float32, device=self.device)
                V = self.criticNN(states).squeeze()
                grad_V = torch.autograd.grad(V.sum(), states, create_graph=True, retain_graph=True)[0]
                xQx = torch.einsum("bi,ij,bj->b", states, self.Q, states)
                uRu = torch.einsum("bi,ij,bj->b", actions, self.R, actions)
                lv_terms = xQx + uRu + (grad_V * Fxu).sum(dim=1)
                self.L_v = (lv_terms**2).mean()
                self.opt_critic.zero_grad()
                self.L_v.backward()
                self.opt_critic.step()
                self.critic_losses.append(float(self.L_v.item()))
                if epoch_iter % self.print_every == 0:
                    print(
                        f"[Phase 1] k={self.k} iter={epoch_iter} | L_v={self.L_v.item():.6f}"
                    )

    def policy_improvement(self):
        L_u = torch.tensor(float("inf"), device=self.device)
        while abs(L_u.item() - self.L_u_prev.item()) > self.epsilon_u:
            self.actorNN.apply(init_weights)
            for epoch_iter in range(int(0.5*self.Nepochs)):
                states = sample_states(self.env, self.batch_size).to(self.device)
                states.requires_grad_(True)
                actions = self.actorNN(states)
                Fxu_np = self.F(
                    states.cpu().detach().numpy(), actions.cpu().detach().numpy()
                )
                Fxu = torch.tensor(Fxu_np, dtype=torch.float32, device=self.device)
                V = self.criticNN(states).squeeze()
                grad_V = torch.autograd.grad(V.sum(), states, create_graph=True)[0]
                xQx = torch.einsum("bi,ij,bj->b", states, self.Q, states)
                uRu = torch.einsum("bi,ij,bj->b", actions, self.R, actions)
                lu_terms = xQx + uRu + (grad_V * Fxu).sum(dim=1)
                L_u = lu_terms.mean()
                self.opt_actor.zero_grad()
                L_u.backward()
                self.opt_actor.step()
                self.actor_losses.append(float(L_u.item()))
                if epoch_iter % self.print_every == 0:
                    print(
                        f"[Phase 2] k={self.k} iter={epoch_iter} | L_u={L_u.item():.6f} | L_u_prev={self.L_u_prev.item():.6f}"
                    )
        self.L_u_prev = L_u

    def run(self):
        self.initialize_critic()
        self.initialize_actor()
        while self.k < self.kmax:
            self.policy_evaluation()
            self.policy_improvement()
            self.k += 1
            print(
                f"End of iteration k={self.k}, L_v={self.L_v.item():.6f}, L_u={self.L_u_prev.item():.6f}"
            )
        return self.actor_losses, self.critic_losses


def trainAlgo2(agent: Algo2):
    raise NotImplementedError()


def train_pirl(agent, manager: Manager, config):
    # TODO: wtf is this function doing
    actor_losses = []
    critic_losses = []

    # PHASE 1 for Algo 2
    if config["algorithm"] == "Algo2":
        print("Starting Phase 1: Admissible Policy Search...")
        P_init = np.eye(agent.state_dim) * 2.0  # Initial Lyapunov guess
        for _ in range(config["init_epochs"]):
            states = manager.get_sobol_samples(config["batch_size"])
            batch = manager.collect_integral_batch(agent, states)
            loss_stab = agent.initialize_policy(batch, P_init)

    # PHASE 2 & 3
    print(f"Starting {config['algorithm']} Main Training...")
    for epoch in range(config["total_epochs"]):
        # Collect Data
        states = manager.get_sobol_samples(config["batch_size"])

        if config["algorithm"] == "Algo2":
            batch = manager.collect_integral_batch(agent, states)
            c_loss, a_loss = agent.update(batch)
        else:
            # Algo 1 Logic (Single step + Dynamics)
            # TODO: this function is not defined
            batch = manager.collect_standard_batch(agent, states)
            c_loss, a_loss = agent.update(batch)

        actor_losses.append(a_loss)
        critic_losses.append(c_loss)

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch} | Critic Loss: {c_loss:.4f} | Actor Loss: {a_loss:.4f}"
            )
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
        if term or trunc:
            break

    states_history = np.array(states_history)

    # 2. Create Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Controller Behavior at Epoch {epoch}")

    # Panel A: State Trajectories
    axes[0].plot(states_history[:, 0], label="cos(theta)")
    axes[0].plot(states_history[:, 1], label="sin(theta)")
    axes[0].plot(states_history[:, 2], label="angular velocity")
    axes[0].set_title("State Convergence")
    axes[0].legend()

    # Panel B: Control Effort (Torque)
    axes[1].plot(actions_history, color="red")
    axes[1].set_title("Control Signal (u)")
    axes[1].set_ylabel("Torque")

    # Panel C: Value Function Heatmap (Fixed angular velocity = 0)
    # This shows if the PINN has learned the "Physics" of the cost
    res = 50
    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(x, y)
    grid_states = torch.tensor(
        np.stack([X.ravel(), Y.ravel(), np.zeros(res * res)], axis=1),
        dtype=torch.float32,
    )

    with torch.no_grad():
        V = agent.critic(grid_states).reshape(res, res).numpy()

    im = axes[2].contourf(X, Y, V, cmap="viridis")
    axes[2].set_title("Critic Value Surface V(x)")
    fig.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.show()
