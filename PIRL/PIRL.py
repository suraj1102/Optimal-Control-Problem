import torch
import torch.nn as nn
import numpy as np


class PIRL:
    def __init__(self, env, config):

        # Make Q, R tensors
        Q = config["cost_matrices"]["Q"]
        R = config["cost_matrices"]["R"]
        Q = torch.diag(torch.tensor(Q))
        R = torch.diag(torch.tensor(R))

        self.Q = Q
        self.R = R
        self.env = env
        self.config = config

        self.instantialize_neural_nets()
        self.instantiate_optimizers()

    def instantialize_neural_nets(self):
        from NeuralNets import Actor, Critic, AdmissibleNet

        self.actor = Actor(self.env, self.config)
        self.critic = Critic(self.env, self.config)
        self.admissible_net = AdmissibleNet(self.env, self.config)

    def instantiate_optimizers(self):
        lr_a = self.config["hyperparameters"]["lr_actor"]
        lr_c = self.config["hyperparameters"]["lr_critic"]
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_c)


class Algo1(PIRL):
    def __init__(self, env, config, system_dynamics_model):
        super().__init__(env, config)
        self.F = system_dynamics_model

    def compute_hamiltonian(self, states, actions):
        states.requires_grad_(True)
        v = self.critic(states)
        grad_v = torch.autograd.grad(v.sum(), states, create_graph=True)[0]
        dx_dt = self.F(states, actions)
        grad_V_dot_f = torch.sum(grad_v * dx_dt, dim=1)  # physics informed part
        current_cost = torch.diag(
            states @ self.Q @ states.T + actions @ self.R @ actions.T
        )
        return grad_V_dot_f + current_cost


class Algo2(PIRL):
    def compute_hamiltonian(self, states, actions, next_states, dt):
        # Integral Hamiltonian
        v_now = self.critic(states)
        v_next = self.critic(next_states)
        cost = (
            torch.diag(states @ self.Q @ states.T + actions @ self.R @ actions.T) * dt
        )
        return cost + v_next.squeeze() - v_now.squeeze()

    def compute_losses(self, batch):
        s, sn, a, dt = batch
        H = self.compute_hamiltonian(s, a, sn, dt)
        loss_bellman = torch.mean(H**2)
        loss_stability = torch.mean(torch.relu(self.critic(sn) - self.critic(s)))
        loss_c = (
            self.config["alpha"] * loss_bellman + self.config["beta"] * loss_stability
        )
        loss_a = torch.mean(self.compute_hamiltonian(s, self.actor(s), sn, dt))
        return {"critic_loss": loss_c, "actor_loss": loss_a}

    def initialize_policy(self, batch, P_matrix):
        s, sn, _, _ = batch
        P = torch.tensor(P_matrix, dtype=torch.float32)

        w_now = torch.diag(s @ P @ s.T)
        w_next = torch.diag(sn @ P @ sn.T)
        loss_stab = torch.mean(torch.relu(w_next - w_now))

        self.opt_a.zero_grad()
        loss_stab.backward()
        self.opt_a.step()

        return loss_stab.item()
