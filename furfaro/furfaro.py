# pinn_value_function.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# --- Neural network (MLP) ---
class PINN(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, activation=nn.Tanh):
        super().__init__()
        num_hidden_units = 128

        self.h1 = nn.Linear(in_dim, num_hidden_units)
        self.y = nn.Linear(num_hidden_units, out_dim)

        self.activation = activation()

        # Initialize weights
        nn.init.uniform_(self.h1.weight)
        nn.init.zeros_(self.y.weight)
        nn.init.uniform_(self.h1.bias)
        nn.init.zeros_(self.y.bias)

    def forward(self, x):
       x = self.h1(x)
       x = self.activation(x)
       x = self.y(x)
       return x


def pde_residual(model, x):
    # x: (N, 2) tensor
    x = x.clone().requires_grad_(True)
    V = model(x) # N, 1

    grads = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True, retain_graph=True)[0]
    # grads is N, 2

    V_x1 = grads[:, 0]
    V_x2 = grads[:, 1]

    x1 = x[:, 0]
    x2 = x[:, 1]
    
    # PDE 
    term1 = torch.square(x1) + torch.square(x2)
    term2 = (-x1 + x2) * V_x1
    term3 = 0.5 * (-x1 - x2 + torch.square(x1) * x2) * V_x2
    term4 = -0.25 * torch.square(x1) * torch.square(V_x2)
    residual = term1 + term2 + term3 + term4

    return residual, V


def sample_inputs(n_sample):
    """
    Generate input mesh for traning (will require grads) or eval
    """
    x1 = np.random.uniform(-1, 1, size=(n_sample, 1))
    x2 = np.random.uniform(-1, 1, size=(n_sample, 1))
    x = np.hstack([x1, x2])
    return torch.tensor(x, dtype=torch.float32, device=device)

def main():
    # ------ Model Training -------
    model = PINN(in_dim=2, out_dim=1, activation=nn.Tanh).to(device)

    # Hyperparameters
    n_colloc = 5000
    lr = 1e-3
    n_epochs = 2000
    pde_weight = 1.0
    bc_weight = 1.0
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Boundary condition point at origin
    x_bc = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=device)
    v_bc = torch.tensor([[0.0]], dtype=torch.float32, device=device)

    # tranining loop
    loss_history = []
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        x_colloc = sample_inputs(n_colloc)
        x_colloc.requires_grad_(True)

        residual, V_out = pde_residual(model, x_colloc)
        pde_loss = torch.mean(residual**2)

        V0_out = model(x_bc)
        bc_loss = torch.mean(V0_out**2)

        loss = pde_weight * pde_loss + bc_weight * bc_loss
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if epoch % 100 == 0:
            print(
                f"Epoch {epoch} | Total Loss: {loss.item():.4e} | "
                f"PDE Loss: {pde_loss.item():.4e} | "
                f"BC Loss: {bc_loss.item():.4e}"
            )

    # Save model
    torch.save(model.state_dict(), "pinn_value_function.pth")
    print("Model saved to pinn_value_function.pth")

if __name__ == "__main__":
    main()
