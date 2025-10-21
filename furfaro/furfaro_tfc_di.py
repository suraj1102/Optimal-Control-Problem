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
        num_hidden_units = 400

        self.h1 = nn.Linear(in_dim, num_hidden_units)
        self.y = nn.Linear(num_hidden_units, out_dim)

        self.activation = activation()

        nn.init.normal_(self.h1.weight)
        nn.init.normal_(self.h1.bias)
        nn.init.normal_(self.y.weight)
        nn.init.zeros_(self.y.bias)

    def forward(self, x):
        x = self.activation(self.h1(x))
        x = self.y(x)
        return x
    
    def initialize_weights(self, X_init):
        # Ensure X_init is a tensor of shape (N_samples, in_dim)
        
        # T is the guess 
        x1 = X_init[:, 0]
        x2 = X_init[:, 1]
        T = 0.5 * (x1 + x2)**2 

        H =  self.activation(X_init @ self.h1.weight.T + self.h1.bias)
        y = H @ self.y.weight.T + self.y.bias

        print(f"""{T.shape=}
{self.h1.weight.shape=} | {self.h1.bias.shape=} 
{self.y.weight.shape=} | {self.y.bias.shape=}
{H.shape=}
{y.shape=}
""")



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
    n_epochs = 5_000

    # Freeze first layer weights and biases
    for param in model.h1.parameters():
        param.requires_grad = False

    x_init = sample_inputs(2000).to(device)
    model.initialize_weights(x_init)

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
