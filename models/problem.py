from pyexpat import model
from utils import *
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from models.hparams import Hyperparams


class problem:
    def __init__(self, hparams: Hyperparams):
        self.hparams = hparams

    def f_x(self,x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def g_x(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def pde_residual(x: torch.Tensor, grad_v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def control_input(self, x: torch.Tensor, grad_v: torch.Tensor) -> torch.Tensor:
        Q = self.hparams.problem_params.Q
        R = self.hparams.problem_params.R
        
        # For inverted pendulum, g_x = [0, 1/(m*l^2)], so g_x^T * ∇V = V_thetadot / (m*l^2)
        # u = -0.5 * R^{-1} * g_x^T * ∇V
        return -0.5 * R[0, 0] * grad_v[:, 1:2] / (self.mass * self.length * self.length)  # grad_v[:, 1] is V_thetadot