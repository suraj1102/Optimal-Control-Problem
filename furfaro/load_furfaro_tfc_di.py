import torch
import numpy as np
import matplotlib.pyplot as plt
from furfaro_tfc_di import PINN  # replace with your PINN class definition file

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# --- Recreate the model (same architecture) ---
model = PINN(in_dim=2, out_dim=1).to(device)

# Load saved weights
model.load_state_dict(torch.load("x_tfc_di.pth", map_location=device))
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
    x_bc = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=device)
    v_bc = torch.tensor([[0.0]], dtype=torch.float32, device=device)
    g = model(X_tensor).cpu().numpy().reshape(nx, nx)
    # convert tensor boundary predictions to numpy scalars for broadcasting with g
    g_0 = model(x_bc).cpu().numpy().squeeze()
    v_bc_val = v_bc.cpu().numpy().squeeze()
    V_pred = g + v_bc_val - g_0

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm

# --- 3D surface plots of V(x1,x2) ---
fig, axes = plt.subplots(1, 3, figsize=(14, 6), subplot_kw={'projection': '3d'})

# Learned solution
surf1 = axes[0].plot_surface(X1, X2, V_pred, cmap=plt.get_cmap("viridis"), linewidth=0, antialiased=True)
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")
axes[0].set_zlabel("V")
axes[0].set_title("Learned V(x1, x2)")
fig.colorbar(surf1, ax=axes[0], shrink=0.6, aspect=10)

# Exact solution
V_exact = exact(X1, X2)
surf2 = axes[1].plot_surface(X1, X2, V_exact, cmap=plt.get_cmap("plasma"), linewidth=0, antialiased=True)
axes[1].set_xlabel("x1")
axes[1].set_ylabel("x2")
axes[1].set_zlabel("V")
axes[1].set_title("Exact V(x1, x2)")
fig.colorbar(surf2, ax=axes[1], shrink=0.6, aspect=10)

# Error surface
V_error = np.abs(V_pred - V_exact)
surf3 = axes[2].plot_surface(X1, X2, V_error, cmap=plt.get_cmap("inferno"), linewidth=0, antialiased=True)
axes[2].set_xlabel("x1")
axes[2].set_ylabel("x2")
axes[2].set_zlabel("Error")
axes[2].set_title("Error |V_pred - V_exact|")
fig.colorbar(surf3, ax=axes[2], shrink=0.6, aspect=10)

# Adjust view
for ax in axes:
    ax.view_init(elev=30, azim=-60)

plt.tight_layout()
plt.show()

