import torch
import torch.nn as nn
import torch.optim as optim
from torch.linalg import pinv
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def sample_inputs(n_sample = 5, dim = 2, middle_weight=0.2):
    """
    Generate input mesh for traning (will require grads) or eval
    """
    xs = []
    for _ in range(dim):
        xi = np.random.uniform(-1, 1, size=(int((1 - middle_weight) * n_sample), 1))
        xi_mid = np.random.uniform(-0.8, 0.8, size=(int(middle_weight * n_sample), 1))
        xi = np.vstack([xi, xi_mid])
        xs.append(xi)
    x = np.hstack(xs)
    return torch.tensor(x, dtype=torch.float32, device=device)

def compute_V_pred_and_exact(model, V_exact_func, n_points=200):
    model.eval()

    nx = n_points
    x1 = np.linspace(-1, 1, nx)
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
    
    return V_pred, V_exact_func(X1, X2), X1, X2