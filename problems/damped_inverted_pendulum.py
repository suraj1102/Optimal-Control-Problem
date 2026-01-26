from models.problem import problem
import numpy as np
import torch
import torch.nn as nn
from models.hparams import Hyperparams

class damped_inverted_pendulum(problem):   
    def __init__(self, hparams: Hyperparams):
        super().__init__(hparams)

        self.hparams.problem_params.labels = ['theta', 'thetadot']

        self.mass = hparams.problem_params.mass_bob
        self.length = hparams.problem_params.length_rod
        self.gravity = hparams.problem_params.gravity
        self.gamma = hparams.problem_params.gamma

    def f_x(self, x: torch.Tensor) -> torch.Tensor:
        theta = x[:, 0]
        theta_dot = x[:, 1]

        return torch.stack([
            theta_dot,
            self.gravity / self.length * torch.sin(theta)
            - (self.gamma / self.mass) * theta_dot
        ], dim=1)


    def g_x(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            torch.zeros_like(x[:, 0], device=x.device),
            torch.ones_like(x[:, 0], device=x.device)
            / (self.mass * self.length ** 2)
        ], dim=1)


    def pde_residual(self, x: torch.Tensor, grad_v: torch.Tensor) -> torch.Tensor:
        # HJB equation residual for inverted pendulum
        x1 = x[:, 0]
        x2 = x[:, 1]
        V_x1 = grad_v[:, 0]
        V_x2 = grad_v[:, 1]

        Q = self.hparams.problem_params.Q
        R = self.hparams.problem_params.R

        raise NotImplementedError()
