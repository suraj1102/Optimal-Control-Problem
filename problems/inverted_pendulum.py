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
        self.eq_point = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=self.hparams.device.device)

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
        x1 = x[:, 0]
        x2 = x[:, 1]
        V_x1 = grad_v[:, 0]
        V_x2 = grad_v[:, 1]

        Q = self.hparams.problem_params.Q
        R = self.hparams.problem_params.R

        term1 = V_x1 * x2 + torch.square(x1) * Q[0, 0] + torch.square(x2) * Q[1, 1] 
        term2 = - torch.square(V_x2) / (4 * self.length**4 * self.mass**2 * R[0, 0])
        term3 = V_x2 * self.gravity * torch.sin(x1) / self.length
        return term1 + term2 + term3
