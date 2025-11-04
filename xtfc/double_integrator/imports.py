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


def compute_V_funcs(model, V_exact_func, n_points=200):
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


def save_model(model, hparams, save_prefix):
    # Save model into file along with hparams txt file
    # Create directory if it doesn't exist
    os.makedirs("./saved_models", exist_ok=True)
    existing = []
    # Check for existing models with same hyperparameters
    for f in os.listdir("saved_models"):
        if f.startswith(save_prefix) and f.endswith("_hparams.txt"):
            hparams_filepath = os.path.join("saved_models", f)
            existing_hparams = {}
            with open(hparams_filepath, 'r') as hpfile:
                for line in hpfile:
                    key, value = line.strip().split(": ", 1)
                    if key == 'activation':
                        existing_hparams[key] = value
                    elif value.isdigit():
                        existing_hparams[key] = int(value)
                    else:
                        try:
                            existing_hparams[key] = float(value)
                        except ValueError:
                            existing_hparams[key] = value
            # Compare all hparams (activation as string)
            match = True
            for key in hparams:
                val = hparams[key]
                if key == 'activation':
                    val = val.__name__
                if key not in existing_hparams or str(existing_hparams[key]) != str(val):
                    match = False
                    break
            if match:
                print(f"Model with identical hyperparameters already exists: {hparams_filepath}. Not saving model.")
                return None, None
    # If no match, save new model
    for f in os.listdir("saved_models"):
        if f.startswith(save_prefix) and f.endswith(".pth"):
            try:
                num = int(f[len(save_prefix):-4])  # strip prefix and ".pth"
                existing.append(num)
            except ValueError:
                continue
    next_num = max(existing) + 1 if existing else 1
    num_str = f"{next_num:03d}"
    filename = f"{save_prefix}{num_str}.pth"
    filepath = os.path.join("saved_models", filename)
    torch.save(model.state_dict(), filepath)
    print(f"Saved model to {filepath}")

    # Save hyperparameters to a text file
    hparams_filename = filename.replace('.pth', '_hparams.txt')
    hparams_filepath = os.path.join("saved_models", hparams_filename)
    with open(hparams_filepath, 'w') as f:
        for key, value in hparams.items():
            if key == 'activation': f.write(f"{key}: {value.__name__}\n")
            else: f.write(f"{key}: {value}\n")
    print(f"Saved hyperparameters to {hparams_filepath}")

    return next_num, filepath