from models.problem import problem
import numpy as np
import torch
import torch.nn as nn
from models.hparams import Hyperparams

class nonlinear_dynamics(problem):
    def __init__(self, hparams: Hyperparams):
        super().__init__(hparams)

    def f_x(self, x: torch.Tensor) -> torch.Tensor:
        # Drift term for double integrator: \dot{x} = [x2, 0] + [0, 1] * u
        return torch.stack([
            -x[:, 0] + x[:, 1],  # \dot{x1} = -x1 + x2
            -0.5 * (x[:, 0] + x[:, 1] - x[:, 0] * x[:, 0] * x[:, 1]) # -0.5 * (x1 + x2 + x1*x1*x2)
        ], dim=1)

    def g_x(self, x: torch.Tensor) -> torch.Tensor:
        # Control matrix: \dot{x} = f_x + g_x * u
        return torch.stack([
            torch.zeros_like(x[:, 0], device=x.device),  # no u in x1_dot
            torch.ones_like(x[:, 0], device=x.device)    # u * x1 term in x2_dot
        ], dim=1)

    def pde_residual(self, x: torch.Tensor, grad_v: torch.Tensor) -> torch.Tensor:
        # HJB equation residual for double integrator
        x1 = x[:, 0]
        x2 = x[:, 1]
        V_x1 = grad_v[:, 0]
        V_x2 = grad_v[:, 1]
        
        term1 = x1 * x1
        term2 = x2 * x2
        term3 = (-x1 + x2) * V_x1
        term4 = (- 0.5 * x1 - 0.5 * x2 + 0.5 * x1 * x1 * x2 ) * V_x2
        term5 = -0.25 * x1 * x1 * V_x2 * V_x2
        return term1 + term2 + term3 + term4 + term5