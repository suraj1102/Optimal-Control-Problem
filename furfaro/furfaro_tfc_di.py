# Furfaro et. al. TFC implementation
"""
TFC:
    V(x, theta) = g(x, theta) + V(0) - g(0, theta)
    g(x, theta) -> NN

Training Steps:
    Randomize and Freeze weights and biases of h1
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.linalg import pinv
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

        nn.init.uniform_(self.h1.weight, -1, 1)
        nn.init.uniform_(self.h1.bias, -1, 1)
        nn.init.uniform_(self.y.weight, -1, 1)
        nn.init.zeros_(self.y.bias)

    def forward(self, x):
        x = self.activation(self.h1(x))
        x = self.y(x)
        return x
    

    def initialize_weights(self, X_init, target_func, regularization=1e-6, device=device):
        X_init = X_init.to(device)

        with torch.no_grad():
            H = self.activation(self.h1(X_init))  # [N, num_hidden_units]
            target = target_func(X_init).to(device)  # [N, out_dim]

            print(f"Hidden layer shape: {H.shape}")
            print(f"Target shape: {target.shape}")

            # Analytical solution: W = (HᵀH + λI)⁻¹ Hᵀ T
            HTH = H.T @ H
            HTT = H.T @ target

            I = torch.eye(HTH.shape[0], device=device)
            HTH_reg = HTH + regularization * I

            try:
                beta_analytical = torch.linalg.solve(HTH_reg, HTT)  # [num_hidden_units, out_dim]

                # Copy to network weights
                self.y.weight.data.copy_(beta_analytical.T)  # PyTorch expects [out_dim, num_hidden_units]
                self.y.bias.data.zero_()

                # Evaluate fit quality
                V_approx = H @ beta_analytical
                mse_error = torch.mean((V_approx - target) ** 2)

                print(f"Analytical initialization completed successfully.")
                print(f"Approximation MSE: {mse_error.item():.6e}")
                print(f"Output weights norm: {torch.norm(self.y.weight).item():.6f}")

                return True, mse_error.item()

            except Exception as e:
                print(f"Analytical solution failed: {e}")
                print("Proceeding with random initialization...")
                return False, float('inf')



def pde_residual(model, x):
    # x: (N, 2) tensor
    x = x.clone().requires_grad_(True)
    x_bc = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=device)
    v_bc = torch.tensor([[0.0]], dtype=torch.float32, device=device)
    
    g = model(x) # N, 1
    g_0 = model(x_bc)

    V = g + v_bc - g_0

    grads = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True, retain_graph=True)[0]
    # grads is N, 2

    V_x1 = grads[:, 0]
    V_x2 = grads[:, 1]

    x1 = x[:, 0]
    x2 = x[:, 1] 

    # PDE terms (same as before but using V from TFC)
    term1 = -0.5 * (torch.square(x1) + torch.square(x2))
    term2 = - V_x1 * x2
    term3 = 0.5 * torch.square(V_x2)
    residual = term1 + term2 + term3

    return residual, V


def sample_inputs(n_sample, middle_weight=0.0):
    """
    Generate input mesh for traning (will require grads) or eval
    """
    x1 = np.random.uniform(-1, 1, size=(int((1 - middle_weight)*n_sample), 1))
    x1_mid = np.random.uniform(-0.8, 0.8, size=(int(middle_weight*n_sample), 1))
    x2 = np.random.uniform(-1, 1, size=((int((1-middle_weight)*n_sample), 1)))
    x2_mid = np.random.uniform(-0.8, 0.8, size=(int(middle_weight*n_sample), 1))

    x1 = np.vstack([x1, x1_mid])
    x2 = np.vstack([x2, x2_mid])
    x = np.hstack([x1, x2])
    return torch.tensor(x, dtype=torch.float32, device=device)


def main():
    # ------ Model Training -------
    model = PINN(in_dim=2, out_dim=1, activation=nn.Tanh).to(device)
    object.__setattr__(model, "debug", True)

    # Hyperparameters
    n_colloc = 5_000
    lr = 5e-3
    n_epochs = 10_001

    # Freeze first layer weights and biases
    for param in model.h1.parameters():
        param.requires_grad = False

    x_init = sample_inputs(2000).to(device)

    target_func = lambda x: 0.5 * torch.square(x[:, 0:1] + x[:, 1:2])
    model.initialize_weights(x_init, target_func)

    # Since weights and biases of h1 are frozen, no need to use model.parameters()
    optimizer = optim.Adam(model.y.parameters(), lr=lr)
    
    # tranining loop
    loss_history = []
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        x_colloc = sample_inputs(n_colloc)

        x_colloc.requires_grad_(True)
        residual, V_out = pde_residual(model, x_colloc)
        pde_loss = torch.mean(residual**2)
        
        loss = pde_loss
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if epoch % 100 == 0:
            print(
                f"Epoch {epoch} | Total Loss: {loss.item():.4e}"
            )

    # Save model
    torch.save(model.state_dict(), "x_tfc_di.pth")

if __name__ == "__main__":
    main()
