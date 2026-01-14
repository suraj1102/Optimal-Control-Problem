from models.problem import problem
import numpy as np
import torch
import torch.nn as nn
from models.hparams import Hyperparams

class double_input_control_problem(problem):
    def __init__(self, hparams: Hyperparams):
        super().__init__(hparams)

    def f_x(self, x: torch.Tensor) -> torch.Tensor:
        # Drift term for double input control problem: \dot{x} = [0, 0] + g_x * u
        return torch.zeros_like(x)

    def g_x(self, x: torch.Tensor) -> torch.Tensor:
        # Control matrix: \dot{x} = f_x + g_x @ u
        # g_x is identity matrix for each sample
        return torch.eye(2, device=x.device).unsqueeze(0).expand(x.shape[0], -1, -1)

    def pde_residual(self, x: torch.Tensor, grad_v: torch.Tensor) -> torch.Tensor:
        # HJB equation residual for double input control problem
        x1 = x[:, 0]
        x2 = x[:, 1]
        V_x1 = grad_v[:, 0]
        V_x2 = grad_v[:, 1]
        
        # Residual: -0.5*(x1^2 + x2^2) + 0.5*(V_x1^2 + V_x2^2)
        term1 = -0.5 * (torch.square(x1) + torch.square(x2))
        term2 = 0.5 * (torch.square(V_x1) + torch.square(V_x2))
        return term1 + term2