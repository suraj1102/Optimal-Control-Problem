from imports import *
from model import *
from train import *
import os
import glob

def exact(x1, x2):
    return np.sqrt(3) / 2 * (x1**2 + x2**2) + x1 * x2

# --- Evaluate on a grid ---
nx = 201
x1 = np.linspace(-1, 1, nx)
x2 = np.linspace(-1, 1, nx)
X1, X2 = np.meshgrid(x1, x2)
X = np.vstack([X1.ravel(), X2.ravel()]).T
X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
x_bc = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=device)
v_bc = torch.tensor([[0.0]], dtype=torch.float32, device=device)

# --- Iterate through all models ---
saved_models_dir = "saved_models"
model_files = glob.glob(os.path.join(saved_models_dir, "*.pth"))

for model_file in model_files:
    # Extract model prefix and number
    model_name = os.path.basename(model_file)
    # Example: di-deep-tfc-50kepochs-027.pth
    # Remove extension
    model_base = os.path.splitext(model_name)[0]
    # Split by '-'
    parts = model_base.split('-')
    # Prefix: everything except last part (number)
    model_prefix = '-'.join(parts[:-1])
    # Number: last part
    model_number = parts[-1]

    # Construct hparams file path
    hparams_file = os.path.join(saved_models_dir, f"{model_prefix}-{model_number}_hparams.txt")

    # Load hyperparameters
    hparams = {}
    with open(hparams_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            key, value = line.split(": ", 1)
            if key == 'activation':
                hparams[key] = getattr(nn, value)  # Convert activation name to function
            elif value.isdigit():
                hparams[key] = int(value)
            else:
                try:
                    hparams[key] = float(value)
                except ValueError:
                    hparams[key] = value

    # Ensure 'hidden_units' is parsed as a list of integers
    if isinstance(hparams['hidden_units'], str):
        hparams['hidden_units'] = [int(x) for x in hparams['hidden_units'].strip('[]').split(',')]

    # Recreate the model
    hidden_units = hparams['hidden_units']
    activation = hparams['activation']
    model = X_TFC(in_dim=2, out_dim=1, hidden_units=hidden_units, activation=activation).to(device)

    # Load saved weights
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    # Evaluate on the grid
    with torch.no_grad():
        g = model(X_tensor).cpu().numpy().reshape(nx, nx)
        g_0 = model(x_bc).cpu().numpy().squeeze()
        v_bc_val = v_bc.cpu().numpy().squeeze()
        V_pred = g + v_bc_val - g_0

    # Exact solution
    V_exact = exact(X1, X2)

    # Error surface
    V_error = np.abs(V_pred - V_exact)

    # Plot the results
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), subplot_kw={'projection': '3d'})

    # Learned solution
    surf1 = axes[0].plot_surface(X1, X2, V_pred, cmap=plt.get_cmap("viridis"), linewidth=0, antialiased=True)
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].set_zlabel("V")
    axes[0].set_title(f"Learned V(x1, x2) - Model {model_number}")
    fig.colorbar(surf1, ax=axes[0], shrink=0.6, aspect=10)

    # Exact solution
    surf2 = axes[1].plot_surface(X1, X2, V_exact, cmap=plt.get_cmap("viridis"), linewidth=0, antialiased=True)
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    axes[1].set_zlabel("V")
    axes[1].set_title("Exact V(x1, x2)")
    fig.colorbar(surf2, ax=axes[1], shrink=0.6, aspect=10)

    # Error surface
    surf3 = axes[2].plot_surface(X1, X2, V_error, cmap=plt.get_cmap("viridis"), linewidth=0, antialiased=True)
    axes[2].set_xlabel("x1")
    axes[2].set_ylabel("x2")
    axes[2].set_zlabel("Error")
    axes[2].set_title("Error |V_pred - V_exact|")
    fig.colorbar(surf3, ax=axes[2], shrink=0.6, aspect=10)

    # Adjust view
    for ax in axes:
        ax.view_init(elev=30, azim=-60)

    # Annotate hyperparameters on the plot
    activation_name = activation.__name__ if hasattr(activation, "__name__") else str(activation)
    hidden_units_str = str(hidden_units)
    fig.suptitle(f"Activation: {activation_name}, Hidden Units: {hidden_units_str}", fontsize=14, y=.92)
    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(saved_models_dir, f"{model_prefix}-{model_number}_plot.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Plots saved for model {model_number} at {plot_file}")
