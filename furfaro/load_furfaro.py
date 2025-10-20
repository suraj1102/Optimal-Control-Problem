import torch
import numpy as np
import matplotlib.pyplot as plt
from furfaro import PINN  # replace with your PINN class definition file

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# --- Recreate the model (same architecture) ---
model = PINN(in_dim=2, out_dim=1).to(device)

# Load saved weights
model.load_state_dict(torch.load("pinn_value_function.pth", map_location=device))
model.eval()
print("Model loaded.")

def exact(x1, x2):
    return 0.5 * x1**2 + x2**2

# --- Evaluate on a grid ---
nx = 201
x1 = np.linspace(-1, 1, nx)
x2 = np.linspace(-1, 1, nx)
X1, X2 = np.meshgrid(x1, x2)
X = np.vstack([X1.ravel(), X2.ravel()]).T
X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

with torch.no_grad():  # NO gradients needed
    V_pred = model(X_tensor).cpu().numpy().reshape(nx, nx)

from matplotlib import cm

# --- 2D Contour plots of V(x1,x2) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot the learned solution
c1 = axes[0].contourf(X1, X2, V_pred, levels=50, cmap=cm.viridis)
fig.colorbar(c1, ax=axes[0], shrink=0.8)
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")
axes[0].set_title("Learned V(x1, x2)")

# Compute the exact solution
V_exact = exact(X1, X2)

# Plot the exact solution
c2 = axes[1].contourf(X1, X2, V_exact, levels=50, cmap=cm.plasma)
fig.colorbar(c2, ax=axes[1], shrink=0.8)
axes[1].set_xlabel("x1")
axes[1].set_ylabel("x2")
axes[1].set_title("Exact V(x1, x2)")

# Compute the error
V_error = np.abs(V_pred - V_exact)

# Plot the error
c3 = axes[2].contourf(X1, X2, V_error, levels=50, cmap=cm.inferno)
fig.colorbar(c3, ax=axes[2], shrink=0.8)
axes[2].set_xlabel("x1")
axes[2].set_ylabel("x2")
axes[2].set_title("Error |V_pred - V_exact|")

plt.tight_layout()
plt.show(block=False)
x = input()
