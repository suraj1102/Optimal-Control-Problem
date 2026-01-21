from pyexpat import model
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
        g_x = self.g_x(x)
        
        u = -0.5 * (torch.inverse(R) @ (g_x @ grad_v.T))
        return u