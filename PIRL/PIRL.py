import torch
import torch.nn as nn
import numpy as np


class PIRL():
    def __init__(self, actor, critic, Q, R, config):
        self.actor = actor
        self.critic = critic
        self.Q = Q
        self.R = R
        self.cfg = config
        
        # Standard optimizers
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=config['lr_a'])
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=config['lr_c'])


class Algo1(PIRL):
    def __init__(self, actor, critic, Q, R, config, system_dynamics):
        super().__init__(actor, critic, Q, R, config)
        self.f = system_dynamics
    def compute_hamiltonian(self, states
    , actions):
        states.requires_grad_(True)
        v = self.critic(states)
        grad_v = torch.autograd.grad(v.sum(), states, create_graph=True)[0]
        dx_dt = self.f(states, actions)
        grad_V_dot_f = torch.sum(grad_v * dx_dt, dim=1)   #physics informed part
        current_cost = torch.diag(states @ self.Q @ states.T + actions @ self.R @ actions.T)
        return grad_V_dot_f + current_cost
    

class Algo2(PIRL):
    def compute_hamiltonian(self, states, actions, next_states, dt):
        #Integral Hamiltonian
        v_now = self.critic(states)
        v_next = self.critic(next_states)
        cost = torch.diag(states @ self.Q @ states.T + actions @ self.R @ actions.T) * dt
        return cost + v_next.squeeze() - v_now.squeeze()
    
    def compute_losses(self, batch):
        s, sn, a, dt = batch
        H = self.compute_hamiltonian(s, a, sn, dt)
        loss_bellman = torch.mean(H**2)
        loss_stability = torch.mean(torch.relu(self.critic(sn) - self.critic(s)))
        loss_c = self.cfg['alpha'] * loss_bellman + self.cfg['beta'] * loss_stability
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