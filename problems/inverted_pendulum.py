from models.problem import problem
import numpy as np
import torch
import torch.nn as nn
from models.hparams import Hyperparams

class inverted_pendulum(problem):   
    def __init__(self, hparams: Hyperparams):
        super().__init__(hparams)

        self.hparams.problem_params.labels = ['theta', 'thetadot']

        self.mass = hparams.problem_params.mass_bob
        self.length = hparams.problem_params.length_rod
        self.gravity = hparams.problem_params.gravity

    def f_x(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            x[:, 1],  # \dot{theta} = thetadot
            self.gravity / self.length * torch.sin(x[:, 0])  # \dot{thetadot} = (g/l) * sin(theta)
        ], dim=1)

    def g_x(self, x: torch.Tensor) -> torch.Tensor:
        # Control matrix: \dot{x} = f_x + g_x * u
        return torch.stack([
            torch.zeros_like(x[:, 0], device=x.device),  # u affects thetadot
            torch.ones_like(x[:, 0], device=x.device) / (self.mass * self.length * self.length)  # u = \dot{thetadot}
        ], dim=1)
    

    def pde_residual(self, x: torch.Tensor, grad_v: torch.Tensor) -> torch.Tensor:
        # HJB equation residual for inverted pendulum
        theta = x[:, 0]
        thetadot = x[:, 1]
        V_theta = grad_v[:, 0]
        V_thetadot = grad_v[:, 1]
        
        # Residual: -0.5*(theta^2 + thetadot^2) - V_theta*thetadot + 0.5*(V_thetadot^2)/(m*l^2) + V_theta * (gravity/l * sin(theta))
        term1 = -0.5 * (torch.square(theta) + torch.square(thetadot))
        term2 = -V_theta * thetadot
        term3 = 0.5 * torch.square(V_thetadot) / (self.mass * self.length * self.length)
        term4 = V_theta * (self.gravity / self.length * torch.sin(theta))
        return term1 + term2 + term3 + term4
