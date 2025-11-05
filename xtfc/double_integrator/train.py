from imports import *
from model import *

import os
from dotenv import load_dotenv
import wandb
import time
from tqdm import tqdm

load_dotenv("/Users/suraj/Library/CloudStorage/OneDrive-PlakshaUniversity/Classes/Sem5/DL/DL-Project/OPC/.env")
KEY = os.getenv("WANDB_API_KEY")
wandb.login(key=KEY)

# Hyperparameters
hparams = {
        'hidden_units': [128],
        'activation': nn.Tanh,
        'n_colloc': 5_000,
        'lr': 1e-3,
        'n_epochs': 50_000,
        'analytical_pretraining': True
    }

# the model save file will have this name as well as the wandb run
# useful to indicate what models were created for what experiments
save_prefix = "di-deep-tfc-50kepochs"

V_guess = lambda x: 0.5 * torch.square(x[:, 0:1] + x[:, 1:2])
V_exact = lambda x1, x2: np.sqrt(3) / 2 * (x1**2 + x2**2) + x1 * x2


pde_loss_history = []
boundary_loss_history = []

# Global variables for tracking model saves and metrics
model_number = None
saved_filename = None
training_time = None

def train(hparams):
    global model_number, saved_filename, training_time
    start_time = time.time()

    # Set these arrays back to empty so that when running multiple models it doesn't keep extending
    global pde_loss_history, boundary_loss_history
    # Reset histories for the current model
    pde_loss_history.clear()
    boundary_loss_history.clear()
    
    hidden_units = hparams['hidden_units']
    activation = hparams['activation']
    n_colloc = hparams['n_colloc']
    lr = hparams['lr']
    n_epochs = hparams['n_epochs']

    model = X_TFC(in_dim=2, out_dim=1, hidden_units=hidden_units, activation=activation).to(device)
    model.train()

    # Freeze layer weights and biases
    for layer in model.layers:
        for p in layer.parameters():
            p.requires_grad = False

    # Ensure output layer is trainable
    for p in model.y.parameters():
        p.requires_grad = True


    if hparams['analytical_pretraining']:
        x_init = sample_inputs(n_sample=2000).to(device)
        model.analytical_pretraning(x_init, V_guess)

    optimizer = optim.Adam(model.y.parameters(), lr=lr)

    progress_bar = tqdm(range(n_epochs), desc="Training Progress", unit="epoch")
    for epoch in progress_bar:
        optimizer.zero_grad()

        # Sample points
        x_colloc = sample_inputs(n_sample=n_colloc)
        x_colloc.requires_grad_(True)

        # Forward pass and residual calulation
        residual, boundry_res, V_out, G_out = pde_residual(model, x_colloc)

        boundry_loss = torch.mean(boundry_res**2)
        pde_loss = torch.mean(residual**2)
        pde_loss.backward()

        optimizer.step()
        pde_loss_history.append(pde_loss.item())
        boundary_loss_history.append(boundry_loss.item())

        progress_bar.set_postfix({
            "PDE Loss": f"{pde_loss.item():.4e}",
            "Boundary Loss": f"{boundry_loss.item():.4e}"
        })

    global model_number, saved_filename, training_time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    model_number, saved_filename = save_model(model, hparams, save_prefix)
    return model, model_number is not None

def log_wandb(model):
    # --- wandb logging ---
    hparams_to_log = hparams.copy()
    if 'activation' in hparams_to_log:
        hparams_to_log['activation'] = hparams_to_log['activation'].__name__

    run = wandb.init(
        project="OPC",
        config=hparams_to_log,
        name=f"{save_prefix}-{model_number}",
        reinit=True,
        resume=False
    )

    # Log training time along with other metrics
    wandb.log({"training_time": training_time})  # in seconds

    for epoch, (pde_loss, boundary_loss) in enumerate(zip(pde_loss_history, boundary_loss_history)):
        wandb.log({
            "pde_loss": float(pde_loss),
            "boundary_loss": float(boundary_loss)
        }, step=epoch)

    # Log Model Performance:
    V_pred, V, X1, X2 = compute_V_funcs(model, V_exact, n_points=200) # 200 x 200
    V_error = np.abs(V_pred - V)
    max_V_error = np.max(V_error)
    avg_V_error = np.mean(V_error)
    print(f"Max V_error: {max_V_error:.4e}, Avg V_error: {avg_V_error:.4e}")

    wandb.log({
        "V_pred": wandb.Image(plt.imshow(V_pred).figure),
        "V_exact": wandb.Image(plt.imshow(V).figure),
        "V_error": wandb.Image(plt.imshow(V_error).figure),
        "X1": wandb.Image(plt.imshow(X1).figure),
        "X2": wandb.Image(plt.imshow(X2).figure),
        "max_V_error": max_V_error,
        "avg_V_error": avg_V_error,
        "saved_filename": saved_filename
    })
    
    plt.close('all')  # Close all figures to free memory

    run.finish()

if __name__ == '__main__':
    activations = [nn.Tanh, nn.ReLU, nn.SiLU, nn.Sigmoid]
    hu_list = [
        [10, 10],
        [10, 50],
        [10, 100],
        [50, 10],
        [50, 50],
        [50, 100],
        [100, 10],
        [100, 50],
        [100, 100]
    ]

    for hu in tqdm(hu_list, desc="Hidden Units Progress", unit="config"):
        hparams['hidden_units'] = hu
        for activation in tqdm(activations, desc="Activations Progress", unit="activation", leave=False):
            hparams['activation'] = activation
            model, is_unique = train(hparams)
            if is_unique:  # if model was unique (not a duplicate)
                log_wandb(model)
            else:
                print("Duplicate Model Found - Skipping wandb logging")