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
    return (np.sqrt(3)/2) * (x1**2 + x2**2) + x1*x2

# --- Evaluate on a grid ---
nx = 201
x1 = np.linspace(-1, 1, nx)
x2 = np.linspace(-1, 1, nx)
X1, X2 = np.meshgrid(x1, x2)
X = np.vstack([X1.ravel(), X2.ravel()]).T
X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

with torch.no_grad():  # NO gradients needed
    V_pred = model(X_tensor).cpu().numpy().reshape(nx, nx)

from mpl_toolkits.mplot3d import Axes3D  # optional in modern matplotlib
from matplotlib import cm

# --- 3D Surface plot of V(x1,x2) ---
fig = plt.figure(figsize=(18, 6))

# Plot the learned solution
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X1, X2, V_pred, cmap=cm.viridis, linewidth=0, antialiased=True)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_zlabel("V(x1,x2)")
ax1.set_title("Learned V(x1, x2) Surface")

# Compute the exact solution
V_exact = exact(X1, X2)

# Plot the exact solution
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X1, X2, V_exact, cmap=cm.plasma, linewidth=0, antialiased=True)
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.set_zlabel("V(x1,x2)")
ax2.set_title("Exact V(x1, x2) Surface")

# Compute the error
V_error = np.abs(V_pred - V_exact)

# Plot the error
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(X1, X2, V_error, cmap=cm.inferno, linewidth=0, antialiased=True)
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
ax3.set_xlabel("x1")
ax3.set_ylabel("x2")
ax3.set_zlabel("Error")
ax3.set_title("Error Surface |V_pred - V_exact|")

plt.tight_layout()
plt.show(block=False)
x = input()
