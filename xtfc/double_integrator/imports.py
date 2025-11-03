import torch
import torch.nn as nn
import torch.optim as optim
from torch.linalg import pinv
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def sample_inputs(n_sample = 2000, dim = 2, middle_weight=0.2):
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