import torch
import torch.nn as nn
from utils import sample_states
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

def plot_value_function(critic: nn.Module):
    import matplotlib.pyplot as plt
    import numpy as np
    
    critic.eval()
    device = next(critic.parameters()).device

    n_theta = 100
    n_thetadot = 100
    theta_range = np.linspace(-np.pi, np.pi, n_theta)
    thetadot_range = np.linspace(-4, 4, n_thetadot)

    Theta, Thetadot = np.meshgrid(theta_range, thetadot_range)

    state_grid = np.stack([Theta, Thetadot], axis=-1)
    state_grid_flat = state_grid.reshape(-1, 2)

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


def plot_action_function(actor: nn.Module):
    import matplotlib.pyplot as plt
    import numpy as np
    
    actor.eval()
    device = next(actor.parameters()).device

    n_theta = 100
    n_thetadot = 100
    theta_range = np.linspace(-np.pi, np.pi, n_theta)
    thetadot_range = np.linspace(-4, 4, n_thetadot)

    Theta, Thetadot = np.meshgrid(theta_range, thetadot_range)

    state_grid = np.stack([Theta, Thetadot], axis=-1)
    state_grid_flat = state_grid.reshape(-1, 2)

    # Evaluate value function
    with torch.no_grad():
        state_tensor = torch.tensor(state_grid_flat, dtype=torch.float32, device=device)
        values = actor(state_tensor).cpu().numpy().reshape(n_thetadot, n_theta)

    # Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Theta, Thetadot, values, cmap='viridis')
    ax.set_xlabel('theta (rad)')
    ax.set_ylabel('thetadot (rad/s)')
    ax.set_zlabel('u')
    ax.set_title('Action Function Surface')
    plt.show()


class Algo1Trainer:
    def __init__(self, agent: Algo1):
        #TODO: add all this to config
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
        self.kmax = int(self.config.get("kmax", 5))
        self.epsilon_v = float(self.config.get("epsilon_v", 1e-2))
        self.epsilon_u = float(self.config.get("epsilon_u", 1e-2))
        self.Nepochs = int(self.config.get("Nepochs", 10_000))
        self.batch_size = int(self.config.get("batch_size", 2**12))
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

    def initalize_actor_lqr(self):
        """
        Initialize the actor network to mimic the LQR policy from linearized dynamics using backpropagation.
        """
        import numpy as np
        import scipy.linalg

        def linearize(F, x0, u0):
            x0 = x0.clone().requires_grad_(True)
            u0 = u0.clone().requires_grad_(True)

            def f_x(x):
                x_b = x.unsqueeze(0)        # [1, state_dim]
                u_b = u0.unsqueeze(0)       # [1, action_dim]
                return F(x_b, u_b).squeeze(0)

            def f_u(u):
                x_b = x0.unsqueeze(0)
                u_b = u.unsqueeze(0)
                return F(x_b, u_b).squeeze(0)

            A = torch.autograd.functional.jacobian(f_x, x0)
            B = torch.autograd.functional.jacobian(f_u, u0)

            return A, B

        # Linearize self.F around the origin: get A, B matrices
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        # Use 1D tensors for x0, u0
        x0 = torch.zeros(state_dim, device=self.device, requires_grad=True)
        u0 = torch.zeros(action_dim, device=self.device, requires_grad=True)

        A, B = linearize(self.F, x0, u0)
        A = A.detach().cpu().numpy()
        B = B.detach().cpu().numpy()

        Q = self.Q.detach().cpu().numpy()
        R = self.R.detach().cpu().numpy()


        # print(f"{x0.shape=}")
        # print(f"{u0.shape=}")
        # print(f"{A.shape=}")
        # print(f"{B.shape=}")
        # print(f"{Q.shape=}")
        # print(f"{R.shape=}")
        # print(f"{x0=}")
        # print(f"{u0=}")
        # print(f"{A=}")
        # print(f"{B=}")
        # print(f"{Q=}")
        # print(f"{R=}")

        # Solve the continuous-time Algebraic Riccati Equation (ARE)
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        K = K.astype("float32") 

        print(f"[LQR INIT] {K=}")

        # Directly set the last layer weights using pseudoinverse to match LQR actions
        self.actorNN.eval()
        with torch.no_grad():
            states = sample_states(self.env, 4096).to(self.device)
            lqr_actions = -torch.from_numpy(K).to(self.device) @ states.T
            lqr_actions = lqr_actions.T  # (batch, action_dim)

            # Forward pass up to last layer
            features = states
            net = self.actorNN.net if hasattr(self.actorNN, 'net') else self.actorNN
            for layer in list(net.children())[:-1]:
                features = layer(features)

            # Last layer
            last_layer = list(net.children())[-1]
            # Solve Wx = u => W = u^T @ pinv(x^T)
            X = features.detach().cpu().numpy()  # (batch, hidden)
            U = lqr_actions.detach().cpu().numpy()  # (batch, action_dim)
            W = U.T @ np.linalg.pinv(X.T)  # (action_dim, hidden)
            b = np.zeros(U.shape[1], dtype=W.dtype)

            last_layer.weight.copy_(torch.from_numpy(W).to(last_layer.weight.device, dtype=last_layer.weight.dtype))
            if last_layer.bias is not None:
                last_layer.bias.copy_(torch.from_numpy(b).to(last_layer.bias.device, dtype=last_layer.bias.dtype))
            print("[LQR INIT] Last layer weights set using pseudoinverse.")

    def policy_evaluation(self):
        self.criticNN.train()
        self.actorNN.eval()

        self.L_v = torch.tensor(float("inf"), device=self.device)
        # while self.L_v.item() > self.epsilon_v:
        self.criticNN.apply(init_weights)
        for epoch_iter in range(self.Nepochs):
            states = sample_states(self.env, self.batch_size).to(self.device)
            states.requires_grad_(True)

            with torch.no_grad():
                actions = self.actorNN(states)

            Fxu = self.F(states, actions)

            # print(f"{states.shape=}\t{actions.shape=}\t{Fxu.shape=}")
            # print(f"{states=}\n{actions=}\n{Fxu=}")

            V = self.criticNN(states).squeeze()
            grad_V = torch.autograd.grad(
                V.sum(), states, create_graph=True, retain_graph=True
            )[0]

            # print(f"{V.shape=}\t{grad_V.shape=}")
            # print(f"{V=}\n{grad_V=}")

            xQx = torch.einsum("bi,ij,bj->b", states, self.Q, states)
            uRu = torch.einsum("bi,ij,bj->b", actions, self.R, actions)
            lv_terms = xQx + uRu + (grad_V * Fxu).sum(dim=1)
            self.L_v = (lv_terms**2).mean()            
            
            
            # Boundary Loss
            # state_dim = self.env.observation_space.shape[0]
            # x0 = torch.zeros(state_dim, device=self.device, requires_grad=True)
            # v0_bc = torch.zeros(1, device=self.device, requires_grad=True)
            # v0_nn = self.criticNN(x0)
            # L_v_bc = (v0_nn-v0_bc)**2
            
            # self.L_v = self.L_v + 10 * L_v_bc.squeeze()

            self.opt_critic.zero_grad()
            self.L_v.backward()
            self.opt_critic.step()
            self.critic_losses.append(float(self.L_v.item()))
            if epoch_iter % self.print_every == 0:
                print(
                    f"[Phase 1] k={self.k} iter={epoch_iter} | L_v={self.L_v.item():.6f}"
                )

    def policy_improvement(self):
        self.actorNN.train()
        self.criticNN.eval()
        for p in self.criticNN.parameters():
            p.requires_grad = False

        L_u = torch.tensor(float("inf"), device=self.device)
        self.actorNN.apply(init_weights)

        for epoch_iter in range(self.Nepochs):
            states = sample_states(self.env, self.batch_size).to(self.device)
            states.requires_grad_(True)

            actions = self.actorNN(states)
            Fxu = self.F(states, actions)

            V = self.criticNN(states).squeeze()
            grad_V = torch.autograd.grad(V.sum(), states, create_graph=True)[0]
            
            xQx = torch.einsum("bi,ij,bj->b", states, self.Q, states)
            uRu = torch.einsum("bi,ij,bj->b", actions, self.R, actions)
            
            lu_terms = xQx + uRu + (grad_V * Fxu).sum(dim=1)
            L_u = lu_terms.mean()
            
            # backprop
            self.opt_actor.zero_grad()
            L_u.backward()
            self.opt_actor.step()
            
            self.actor_losses.append(float(L_u.item()))
            if epoch_iter % self.print_every == 0:
                print(
                    f"[Phase 2] k={self.k} iter={epoch_iter} | L_u={L_u.item():.6f} | L_u_prev={self.L_u_prev.item():.6f}"
                )

            # if self.L_u_prev.item() < float("inf") and torch.abs(L_u - self.L_u_prev).item() <= self.epsilon_u:
            #     break

        self.L_u_prev = L_u

        for p in self.criticNN.parameters():
            p.requires_grad = True
        self.criticNN.train()

    def run(self):
        self.initalize_actor_lqr()
        plot_action_function(self.actorNN)

        while self.k < self.kmax:
            
            self.policy_evaluation()
            plot_value_function(self.criticNN)

            self.policy_improvement()
            plot_action_function(self.actorNN)

            self.k += 1
            print(
                f"End of iteration k={self.k}, L_v={self.L_v.item():.6f}, L_u={self.L_u_prev.item():.6f}"
            )
        return self.actor_losses, self.critic_losses


def trainAlgo2(agent: Algo2):
    raise NotImplementedError()
