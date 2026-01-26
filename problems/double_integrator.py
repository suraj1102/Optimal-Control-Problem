from models.problem import problem
import numpy as np
import torch
import torch.nn as nn
from models.hparams import Hyperparams

class double_integrator(problem):
    def __init__(self, hparams: Hyperparams):
        super().__init__(hparams)

        self.hparams.problem_params.labels = ['position', 'velocity']

    def f_x(self, x: torch.Tensor) -> torch.Tensor:
        # Drift term for double integrator: \dot{x} = [x2, 0] + [0, 1] * u
        return torch.stack([
            x[:, 1],  # \dot{x1} = x2
            torch.zeros_like(x[:, 0], device=x.device)  # \dot{x2} = 0 (drift)
        ], dim=1)

    def g_x(self, x: torch.Tensor) -> torch.Tensor:
        # Control matrix: \dot{x} = f_x + g_x * u
        return torch.stack([
            torch.zeros_like(x[:, 0], device=x.device),  # u affects x2
            torch.ones_like(x[:, 0], device=x.device)    # u = \dot{x2}
        ], dim=1)

    def pde_residual(self, x: torch.Tensor, grad_v: torch.Tensor) -> torch.Tensor:
        # HJB equation residual for double integrator
        x1 = x[:, 0]
        x2 = x[:, 1]
        V_x1 = grad_v[:, 0]
        V_x2 = grad_v[:, 1]
        
        # Residual: -0.5*(x1^2 + x2^2) - V_x1*x2 + 0.5*V_x2^2
        term1 = -0.5 * (torch.square(x1) + torch.square(x2))
        term2 = -V_x1 * x2
        term3 = 0.5 * torch.square(V_x2)
        return term1 + term2 + term3