from utils import *
from xtfc_model import ValueFunctionModel
from hparams import hparams
from pinn_model import Pinn

import os
from dotenv import load_dotenv
import wandb
import time
from tqdm import tqdm

LOG_WANDB = False
SHOW_TEST_PLOT = True

V_guess = None
V_exact = None

m = hparams.get('mass', 1)
m_cart = hparams.get('mass_cart', 1)
cart_height = hparams.get('cart_height', 0.4)
l = hparams.get('length', 1)
gravity = hparams.get('gravity', 9.81)

# Properly initialize pde_residual with a default placeholder function - for LSP and autocomplete
def default_pde_residual(x: torch.Tensor, grad_v: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x[:, 0])

compute_pde_residual = default_pde_residual

def default_control_input(x: torch.Tensor, grad_v: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x[:, 0])

compute_control_input = default_pde_residual


def start_wandb():
    load_dotenv("/Users/suraj/Library/CloudStorage/OneDrive-PlakshaUniversity/Classes/Sem5/DL/DL-Project/OPC/.env")
    KEY = os.getenv("WANDB_API_KEY")
    wandb.login(key=KEY)


def start_wandb_run():
    hparams_to_log = hparams.copy()
    if 'activation' in hparams_to_log:
        hparams_to_log['activation'] = hparams_to_log['activation'].__name__

    run = wandb.init(
        project="OPCTest",
        config=hparams_to_log,
        name=f"{hparams['problem']}-{hparams['architecture']}-{hparams_to_log['activation']}-{time.strftime('%d%m%Y-%H%M%S')}",
        reinit=True,
        resume=False
    )
    return run


def set_problem_parameters():
    """
    Sets the parameters for a specific problem type.
    These include the V_guess, V_exact, and pde_residual functions.
    """
    global V_exact, V_guess, compute_pde_residual, compute_control_input
    problem = hparams['problem']

    if hparams['analytical_pretraining'] in ['None', None]:
        V_guess = None

    if problem == 'double-integrator':
        V_exact = lambda x1, x2: np.sqrt(3) / 2 * (x1**2 + x2**2) + x1 * x2

        if hparams['analytical_pretraining'] == 'xTQx':
            V_guess = lambda x: 0.5 * torch.square(x[:, 0:1] + x[:, 1:2])
        elif hparams['analytical_pretraining'] == 'LQR':
            pass # TODO

        def pde_residual_di(x: torch.Tensor, grad_v: torch.Tensor):
            x1 = x[:, 0]
            x2 = x[:, 1]
            V_x1 = grad_v[:, 0]
            V_x2 = grad_v[:, 1]
            term1 = -0.5 * (torch.square(x1) + torch.square(x2))
            term2 = - V_x1 * x2
            term3 = 0.5 * torch.square(V_x2)
            return term1 + term2 + term3
        
        compute_pde_residual = pde_residual_di

        def control_input_di(x: torch.Tensor, grad_v: torch.Tensor) -> torch.Tensor:
            Q = torch.tensor(np.asarray(hparams['Q']), device=device)
            R = torch.tensor(np.asarray(hparams['R']), device=device)

            f_x = torch.stack([
                x[:, 1],
                torch.zeros_like(x[:, 0], device=device),
            ], dim=1)

            g_x = torch.stack([
                torch.zeros_like(x[:, 0], device=device),
                torch.ones_like(x[:, 0], device=device)
            ], dim=1)

            grad_v = grad_v.to(device)
            return -0.5 * R @ (g_x @ grad_v.T)
        
        compute_control_input = control_input_di

    elif problem == 'nonlinear-dynamics':
        pass # TODO

    elif problem == 'inverted-pendulum':
        V_exact = lambda x1, x2: 0.0
        if hparams['analytical_pretraining'] == 'xTQx':
            V_guess = lambda x: 0.5 * torch.square(x[:, 0:1] + x[:, 1:2])
        elif hparams['analytical_pretraining'] == 'LQR':
            pass # TODO

        def pde_residual_ip(x: torch.Tensor, grad_v: torch.Tensor):
            x1 = x[:, 0]
            x2 = x[:, 1]
            V_x1 = grad_v[:, 0]
            V_x2 = grad_v[:, 1]

            Q = hparams['Q']
            R = hparams['R']

            term1 = V_x1 * x2 + torch.square(x1) * Q[0, 0] + torch.square(x2) * Q[1, 1] 
            term2 = - torch.square(V_x2) / (4 * l**4 * m**2 * R[0, 0])
            term3 = V_x2 * gravity * torch.sin(x1) / l
            return term1 + term2 + term3

        compute_pde_residual = pde_residual_ip

    elif problem == 'double-input-cart-pole':
        V_exact = lambda x1, x2: 0

        def control_input_dicp(x: torch.Tensor, grad_v: torch.Tensor) -> torch.Tensor:
            Q = torch.tensor(np.asarray(hparams['Q']), device=device)
            R = torch.tensor(np.asarray(hparams['R']), device=device)

            f_x = torch.stack([
                x[:, 1],
                torch.zeros_like(x[:, 0], device=device),
                x[:, 3],
                -(gravity / l) * torch.sin(x[:, 2])
            ], dim=1) 

            g_x = torch.tensor([ 
                [torch.zeros_like(x[:, 0], device=device), torch.zeros_like(x[:, 0], device=device)],
                [torch.ones_like(x[:, 0], device=device) / m_cart, torch.zeros_like(x[:, 0], device=device)],
                [torch.zeros_like(x[:, 0], device=device), torch.zeros_like(x[:, 0], device=device)],
                [torch.ones_like(x[:, 0], device=device) * cart_height / (-m * l * l), torch.ones_like(x[:, 0], device=device) / (-m * l * l)]
            ])

            grad_v = grad_v.to(device)


            return -0.5 * R @ (g_x @ grad_v.T)
        
        compute_control_input = control_input_dicp

        if hparams['analytical_pretraining'] == 'xTQx':
            Q = torch.tensor(np.asarray(hparams['Q']), device=device)
            V_guess = lambda x: torch.matmul(torch.matmul(x, Q.to(torch.float32)), x.T)
        elif hparams['analytical_pretraining'] == 'LQR':
            pass # TODO

        def pde_residual_ip(x: torch.Tensor, grad_v: torch.Tensor):
                x1 = x[:, 0]
                x2 = x[:, 1]
                x3 = x[:, 2]
                x4 = x[:, 3]
                V_x1 = grad_v[:, 0]
                V_x2 = grad_v[:, 1]
                V_x3 = grad_v[:, 2]
                V_x4 = grad_v[:, 3]

                Q = hparams['Q']
                R = hparams['R']
                q11 = Q[0, 0]
                q22 = Q[1, 1]
                q33 = Q[2, 2]
                q44 = Q[3, 3]
                r11 = R[0, 0]
                r22 = R[1, 1]

                term1 = q11 * x1**2
                term2 = - (1 / (4 * m_cart**2 * r11)) * (V_x2*2)
                term3 = + (cart_height / (2 * l**2 * m_cart * m * r11)) * (V_x2 * V_x4)   # minus minus = plus
                term4 = + x4 * V_x3   # minus(-x4*V_x3)
                term5 = - (1/(4*l**4*m**2*r22) + cart_height**2/(4*l**4*m**2*r11)) * (V_x4*2)
                term6 = - (gravity * torch.sin(x3) / l) * V_x4
                term7 = + x2 * V_x1                                          # minus(-x2*V_x1)
                term8 = q22 * x2**2
                term9 = q33 * x3**2
                term10 = q44 * x4**2

                return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10

        compute_pde_residual = pde_residual_ip

    else:
        raise ValueError(f"Unknown Problem '{problem}' entered")
    

def set_optimizer_scheduler(model: ValueFunctionModel):
    optimizer_name = hparams.get('optimizer', 'ADAM')
    lr = hparams.get('lr', 1e-3)

    if 'unfreeze' in hparams.get('architecture', 'xtfc').lower():
        model_params = model.parameters()
    elif 'pinn' in hparams.get('architecture', 'xtfc').lower():
        model_params = model.parameters()
    else:
        model_params = model.y.parameters()

    # Select optimizer
    if optimizer_name in ['ADAM', 'Adam', 'adam', torch.optim.Adam, optim.Adam]:
        optimizer = optim.Adam(model_params, lr=lr)
    elif optimizer_name in ['LBFGS', 'lbfgs', torch.optim.LBFGS, optim.LBFGS]:
        optimizer = optim.LBFGS(model_params, lr=lr)
    elif callable(optimizer_name):
        opt_instance = optimizer_name(model_params, lr=lr)
        if not isinstance(opt_instance, torch.optim.Optimizer):
            raise TypeError("Custom optimizer must be a subclass of torch.optim.Optimizer")
        optimizer = opt_instance
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Select scheduler
    scheduler_name = hparams.get('Scheduler', 'None')
    scheduler = None
    if scheduler_name in ['exponential', 'ExponentialLR', torch.optim.lr_scheduler.ExponentialLR, optim.lr_scheduler.ExponentialLR]:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("Optimizer must be a torch.optim.Optimizer for scheduler")
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=hparams.get('gamma', 0.99), last_epoch=hparams.get('n_epochs', 10_000))
    elif scheduler_name in ['reduce-on-plateau', 'ReduceLROnPlateau', torch.optim.lr_scheduler.ReduceLROnPlateau, optim.lr_scheduler.ReduceLROnPlateau]:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=hparams.get('patience', 100))
    elif scheduler_name in ['cosine-annealing', 'CosineAnnealingLR', torch.optim.lr_scheduler.CosineAnnealingLR, optim.lr_scheduler.CosineAnnealingLR]:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hparams.get('T_max', 100))
    elif scheduler_name in ['adaptive', 'None', None]:
        scheduler = None
    elif callable(scheduler_name):
        scheduler_instance = scheduler_name(optimizer)
        if scheduler_instance is not None and not isinstance(scheduler_instance, torch.optim.lr_scheduler._LRScheduler):
            raise TypeError("Custom scheduler must be a subclass of torch.optim.lr_scheduler._LRScheduler")
        scheduler = scheduler_instance
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return optimizer, scheduler


def set_parameter_freezing(model: ValueFunctionModel):
    architecture = hparams['architecture'].lower()
    if 'unfreeze' in architecture or 'pinn' in architecture:
        model.unfreeze_parameters('all')
    else:
        model.freeze_parameters('layers')
        model.unfreeze_parameters('output')


def early_stopping(pde_loss: float, patience: int = hparams['early_stopping']) -> bool:
    """
    Implements early stopping logic using a static variable to track the best loss.

    Args:
        pde_loss (float): Current PDE loss.
        patience (int): Number of epochs to wait for improvement before stopping.

    Returns:
        bool: True if training should stop, False otherwise.
    """
    if patience <= 0: # If early stopping is turned off
        return False
    
    if not hasattr(early_stopping, "best_loss"):
        early_stopping.best_loss = float('inf')
        early_stopping.epochs_without_improvement = 0

    if pde_loss < early_stopping.best_loss:
        early_stopping.best_loss = pde_loss
        early_stopping.epochs_without_improvement = 0
    else:
        early_stopping.epochs_without_improvement += 1

    return early_stopping.epochs_without_improvement >= patience


def save_model(model: torch.nn.Module, filename: str, hparams):
    activation_name = hparams['activation'].__name__ if hasattr(hparams['activation'], "__name__") else str(hparams['activation'])
    hidden_units_str = str(hparams['hidden_units'])
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    filename = os.path.join(models_dir, f"{hparams['problem']}_{hparams['architecture']}_{hidden_units_str}_{activation_name}.pt")
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


def train(hparams=hparams):
    set_problem_parameters()

    if 'pinn' in hparams['architecture'].lower():
        is_pinn = True
    else:
        is_pinn = False

    if is_pinn:
        model = Pinn(in_dim=hparams['in_dim'], out_dim=hparams['out_dim'], hparams=hparams).to(device)
    else:
        model = ValueFunctionModel(in_dim=hparams['in_dim'], out_dim=hparams['out_dim'], hparams=hparams).to(device)

    model.train()

    if hparams['log_wandb']:
        run = start_wandb_run()

    if hparams['analytical_pretraining'] and V_guess is None:
        raise ValueError("V_guess is not defined for the selected analytical pretraining method.")

    if hparams['analytical_pretraining'] == 'xTQx':
        model.xTQx_analytical_pretraning(V_guess)

    if not is_pinn:
        set_parameter_freezing(model)

    optimizer, scheduler = set_optimizer_scheduler(model)

    progress_bar = tqdm(range(hparams['n_epochs']), desc="Training Progress", unit="epoch")
    for epoch in progress_bar:
        optimizer.zero_grad()

        # Sample points
        x_colloc = sample_inputs(n_sample=hparams['n_colloc'])
        x_colloc.requires_grad_(True)

        # Forward pass and residual calculation
        g_x, g_0, v, grad_v = model.get_outputs(x_colloc)
        pde_residual = compute_pde_residual(x_colloc, grad_v)
        boundary_residual = model.v_bc - g_0

        boundary_loss = torch.mean(boundary_residual**2)
        pde_loss = torch.mean(pde_residual**2)

        if is_pinn:
            pde_loss = pde_loss + boundary_loss

        pde_loss.backward()

        if isinstance(optimizer, torch.optim.LBFGS):
            optimizer.step(lambda: pde_loss)  # Closure for LBFGS
        else:
            optimizer.step()

        # Step the scheduler if it exists
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(pde_loss)  # ReduceLROnPlateau requires the loss value
            else:
                scheduler.step()

        progress_bar.set_postfix({
            "PDE Loss": f"{pde_loss.item():.4e}",
            "Boundary Loss": f"{boundary_loss.item():.4e}",
            "LR": f"{optimizer.param_groups[0]['lr']:.4e}"
        })

        # Log losses to wandb inside the train function
        if hparams['log_wandb']:
            run.log({
                "pde_loss": float(pde_loss.item()),
                "boundary_loss": float(boundary_loss.item()),
            })

        # Early stopping check
        if hparams['early_stopping'] > 0:
            if early_stopping(pde_loss.item()):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                if hparams['log_wandb']:
                    run.log({"early_stopping_epoch": epoch + 1})

                break

    if hparams['save_model']:
        save_model(model, None, hparams)
    
    return model, (run if LOG_WANDB else None), pde_loss.item(), boundary_loss.item()


def test(model: torch.nn.Module, run: wandb.Run | None, hparams):
    model.eval()
    plt.close('all')

    V_pred, V, X1, X2 = compute_V_pred_and_exact(model, V_exact, n_points=200, hparams=hparams) # 200 x 200
    if V_pred is not None and V is not None and X1 is not None and X2 is not None:
        V_error = np.abs(V_pred - V)
        max_V_error = np.max(V_error)
        avg_V_error = np.mean(V_error)
        print(f"Max V_error: {max_V_error:.4e}, Avg V_error: {avg_V_error:.4e}")

        # Log metrics to wandb
        if run:
            run.log({
                "max_V_error": max_V_error,
                "avg_V_error": avg_V_error,
            })

        # Plot the results
        fig, axes = plt.subplots(1, 3, figsize=(14, 6), subplot_kw={'projection': '3d'})

        # Learned solution
        surf1 = axes[0].plot_surface(X1, X2, V_pred, cmap=plt.get_cmap("viridis"), linewidth=0, antialiased=True)
        axes[0].set_xlabel("x1")
        axes[0].set_ylabel("x2")
        axes[0].set_zlabel("V")
        axes[0].set_title("Learned V(x1, x2)")
        fig.colorbar(surf1, ax=axes[0], shrink=0.6, aspect=10)

        # Exact solution
        surf2 = axes[1].plot_surface(X1, X2, V, cmap=plt.get_cmap("viridis"), linewidth=0, antialiased=True)
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
        activation_name = hparams['activation'].__name__ if hasattr(hparams['activation'], "__name__") else str(hparams['activation'])
        hidden_units_str = str(hparams['hidden_units'])
        run_id = run.id if run else 'local' # Retrieve the run ID from wandb
        fig.suptitle(f"Run ID: {run_id}\nActivation: {activation_name}, Hidden Units: {hidden_units_str}\nMax V_error: {max_V_error:.4e}, Avg V_error: {avg_V_error:.4e}", fontsize=14, y=.92)
        plt.tight_layout()

        # Save the plot as an image file
        plot_filename = f"plot_{run.id if run else 'local'}_{activation_name}_{hidden_units_str}.png"
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_filename}")

        # Log the plot to wandb
        if run:
            run.log({
                "V_pred": wandb.Image(fig),
            })

        if hparams['plot_graphs']:
            print("Displaying plot (plot_graphs=True)...")
            plt.show(block=False)
            plt.pause(3)  # Allow the plot window to update
        else:
            plt.close('all')  # Close all figures to free memory

    if run:
        run.finish()


def test_pendulum_stability(model: ValueFunctionModel, inital_conditions=[0.1, 0.1]):
    initial_states = torch.tensor([inital_conditions], device=device) # x1, x2 = theta, dot(theta)
    
    # Control Loop
    x = initial_states
    time_horizon = 1000  # Set a large time horizon
    dt = 0.01  # Time step for simulation
    trajectory = [x.clone().detach()]  # Store the trajectory for analysis
    u_vals = []  # Store the trajectory for analysis

    for _ in range(time_horizon):
        g_x, g_0, v, grad_v = model.get_outputs(x)
        control_input = compute_control_input(x, grad_v)
        
        f_x = torch.stack([
            x[:, 1],
            gravity / l * torch.sin(x[:, 0])
        ], dim=1)

        g_x = torch.stack([
            torch.zeros_like(x[:, 0], device=device),
            torch.ones_like(x[:, 0], device=device) / (m * l * l)
        ], dim=1)

        x_dot = f_x + g_x * control_input
        x = x + x_dot * dt  # Update state using Euler integration

        # Wrap angle x1 to [-pi, pi] so that x=2pi is equivalent to x=0
        x[:, 0] = (x[:, 0] + np.pi) % (2 * np.pi) - np.pi

        trajectory.append(x.clone().detach())
        u_vals.append(control_input.clone().detach())

    trajectory = torch.stack(trajectory)  # Convert trajectory to a tensor for analysis
    u_vals = torch.stack(u_vals)
    # Plot the trajectory
    trajectory = trajectory.cpu().numpy()  # Convert to numpy for plotting
    u_vals = u_vals.cpu().numpy()
    time_steps = np.arange(len(trajectory)) * dt

    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, trajectory[:, 0, 0], label="x1 (Position)", color="tab:blue")
    plt.plot(time_steps, trajectory[:, 0, 1], label="x2 (Velocity)", color="tab:orange")
    plt.plot(time_steps[:-1], u_vals[:, 0, 0], label="u", color="tab:red")
    plt.xlabel("Time (s)")
    plt.ylabel("State")
    plt.title("Time vs Trajectory")
    plt.legend()
    plt.grid()
    plt.show()

    print("Final State:", trajectory[-1, 0, :])


def test_double_integrator_stability(model: ValueFunctionModel, no_input=False):
    initial_states = torch.tensor([[0.3, 0.3]], device=device) # x1, x2 = theta, dot(theta)
    
    # Control Loop
    x = initial_states
    time_horizon = 1000  # Set a large time horizon
    dt = 0.001  # Time step for simulation
    trajectory = [x.clone().detach()]  # Store the trajectory for analysis

    for _ in range(time_horizon):
        g_x, g_0, v, grad_v = model.get_outputs(x)
        control_input = compute_control_input(x, grad_v)
        
        f_x = torch.stack([
            x[:, 1],
            torch.zeros_like(x[:, 0], device=device)
        ], dim=1)

        g_x = torch.stack([
            torch.zeros_like(x[:, 0], device=device),
            torch.ones_like(x[:, 0], device=device) / (m * l * l)
        ], dim=1) 

        x_dot = f_x + g_x * control_input
        x = x + x_dot * dt  # Update state using Euler integration
        trajectory.append(x.clone().detach())

    trajectory = torch.stack(trajectory)  # Convert trajectory to a tensor for analysis
    # Plot the trajectory
    trajectory = trajectory.cpu().numpy()  # Convert to numpy for plotting
    time_steps = np.arange(len(trajectory)) * dt

    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, trajectory[:, 0, 0], label="x1 (Position in rads)")
    plt.plot(time_steps, trajectory[:, 0, 1], label="x2 (Velocity in rads/s)")
    plt.xlabel("Time (s)")
    plt.ylabel("State")
    plt.title("Time vs Trajectory")
    plt.legend()
    plt.grid()
    plt.show()

# Fixing the train function call in the main block
if __name__ == '__main__':
    if LOG_WANDB:
        start_wandb()

    # activations = [nn.SiLU]
    # hu_list = [
    #     [30],
    # ]

    # for hu in tqdm(hu_list, desc="Hidden Units Progress", unit="config"):
    #     hparams['hidden_units'] = hu
    #     for activation in tqdm(activations, desc="Activations Progress", unit="activation", leave=False):
    # hparams['activation'] = activation

    LOAD = hparams['LOAD_MODEL']

    if not LOAD:
        model, run, _, _ = train(hparams)
    else:
        activation_name = hparams['activation'].__name__ if hasattr(hparams['activation'], "__name__") else str(hparams['activation'])
        hidden_units_str = str(hparams['hidden_units'])
        models_dir = "models"
        filename = os.path.join(models_dir, f"{hparams['problem']}_{hparams['architecture']}_{hidden_units_str}_{activation_name}.pt")
        
        if 'pinn' in hparams['architecture'].lower():
            model = Pinn(in_dim=2, out_dim=1, hparams=hparams).to(device)
        else:
            model = ValueFunctionModel(in_dim=2, out_dim=1, hparams=hparams).to(device)
        
        model.load_state_dict(torch.load(filename, map_location=device))
        run = None
        print(f"Loaded model from {filename}")

    set_problem_parameters()
        
    if run:
        print("Going into Test")
    test(model, run, hparams)
    if hparams['problem'] == 'inverted-pendulum':
        test_pendulum_stability(model, [0.1, 0.3])
        pass
    elif hparams['problem'] == 'double-integrator':
        test_double_integrator_stability(model)
