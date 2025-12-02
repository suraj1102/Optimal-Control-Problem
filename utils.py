import torch
import torch.nn as nn
import torch.optim as optim
from torch.linalg import pinv
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import argparse
import os

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def sample_inputs(n_sample = 5, dim = 2, edge_weight = 0.2, input_range = (-1, 1)):
    """
    Generate input mesh for traning (will require grads) or eval
    """
    xs = []

    n_edge = int(edge_weight * n_sample / 2)
    n_mid = int((1 - edge_weight) * n_sample)

    for _ in range(dim):
        xi_edge_1 = np.random.uniform(input_range[0], 0.8 * input_range[0], size=(n_edge, 1))
        xi_edge_2 = np.random.uniform(0.8 * input_range[1], input_range[1], size=(n_edge, 1))
        xi_mid = np.random.uniform(0.8 * input_range[0], 0.8 * input_range[1], size=(n_mid, 1))
        xi = np.vstack([xi_edge_1, xi_edge_2, xi_mid])
        # xi = np.random.uniform(input_range[0], input_range[1], size=(n_sample, 1))
        xs.append(xi)
    x = np.hstack(xs)
    return torch.tensor(x, dtype=torch.float32, device=device)

def compute_V_pred_and_exact(model, V_exact_func, n_points=200, hparams=None):
    model.eval()

    nx = n_points
    x1 = np.linspace(hparams['input_range'][0], hparams['input_range'][1], nx)
    x2 = np.linspace(-1, 1, nx)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.vstack([X1.ravel(), X2.ravel()]).T
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    with torch.no_grad():  # NO gradients needed
        x_bc = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=device)
        v_bc = torch.tensor([[0.0]], dtype=torch.float32, device=device)
        g = model(X_tensor).cpu().numpy().reshape(nx, nx)
        # convert tensor boundary predictions to numpy scalars for broadcasting with g
        g_0 = model(x_bc).cpu().numpy().squeeze()
        v_bc_val = v_bc.cpu().numpy().squeeze()
        V_pred = g + v_bc_val - g_0
    
    V_exact = V_exact_func(X1, X2) if V_exact_func is not None else None
    return V_pred, V_exact, X1, X2


class SinAct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x