from pyexpat import model
from utils import *
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch

#TODO: implement multiple input ranges logic
#      complete test stability logic in child classes
#      dicp problem entirely  

class problem:
    def __init__(self):
        #----------------------------------
        # hyperparameters
        self.problem = None
        self.architecture = None          # xtfc, xtfc-w-bias, xtfc-unfreeze, pinn
        self.analytical_pretraining = None # None, xTQx, LQR (yet to be implemented)
        self.in_dim = None
        self.out_dim = None
        self.hidden_units = None
        self.activation = None
        self.n_colloc = None
        self.input_range = None
        self.edge_sampling_weight = None
        self.lr = None
        self.optimizer = None            # ADAM or LBFGS (TODO:research LBFGS)
        self.Scheduler = None            # ignored scheduler and patience for now(TODO:research schedulers better)
        self.patience = None             # For reduce-on-plateau scheduler
        self.gamma = None                # For exponential scheduler
        self.n_epochs = None
        self.early_stopping = None       # Indicates patience (in no. of epochs), -1 means no early stopping
        self.log_wandb = None
        self.plot_graphs = None
        self.save_model = None
        self.model_save_path = None
        self.save_plot = None
        self.initial_conditions = None    #for testing stability 

        self.Q = None
        self.R = None
  
        self.mass = None
        self.mass_cart = None
        self.cart_height = None
        self.length = None
        self.gravity = None
        self.LOAD_MODEL = None

        #----------------------------------
        #Value function 
        self.V_exact = None         # defined in child class if available
        self.V_guess = None         # none until proven xTQx (or LQR etc.) 

        #----------------------------------
        #system dynamics and pde residual

        def f_x(self,x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

        def g_x(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

        def control_input(x: torch.Tensor, grad_v: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x[:, 0])
        
        def pde_residual(x: torch.Tensor, grad_v: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x[:, 0])

        self.compute_pde_residual = pde_residual
        self.compute_control_input = control_input

    def simulate_trajectory(self, model, initial_conditions, f_x_func, g_x_func, time_horizon=1000, dt=0.01, angle_wrap_indices=None):
            """
            Args:
                model: Trained model
                initial_conditions: List of initial state values
                f_x_func: Function that takes x (tensor) and returns f_x (drift term)
                g_x_func: Function that takes x (tensor) and returns g_x (control matrix)
                time_horizon: Number of simulation steps
                dt: Time step
                angle_wrap_indices: List of state indices to wrap (e.g., [0] for pendulum angle)
            
            Returns:
                trajectory: Tensor of shape (time_horizon+1, in_dim)
            """
            model.eval()
            initial_states = torch.tensor([initial_conditions], dtype=torch.float32, device=device)
            x = initial_states
            trajectory = [x.clone().detach()]
            
            for _ in range(time_horizon):
                g_x_model, g_0, v, grad_v = model.get_outputs(x)
                control_input = self.compute_control_input(x, grad_v)
                
                f_x = self.f_x(x)
                g_x = self.g_x(x)
                
                # Compute x_dot
                if g_x.dim() == 2:  # Vector g_x (single control)
                    x_dot = f_x + g_x * control_input
                elif g_x.dim() == 3:  # Matrix g_x (multiple controls)
                    x_dot = f_x + g_x @ control_input
                else:
                    raise ValueError("g_x must be 2D (vector) or 3D (matrix)")
                
                x = x + x_dot * dt
                
                # Angle wrapping if specified
                if angle_wrap_indices:
                    for idx in angle_wrap_indices:
                        x[:, idx] = (x[:, idx] + np.pi) % (2 * np.pi) - np.pi
                
                trajectory.append(x.clone().detach())
            
            trajectory = torch.stack(trajectory)
            return trajectory.squeeze(1)  # Shape: (time_horizon+1, in_dim)
                

    def test_stability(self, trajectory, dt=0.01, state_labels=None, title="Stability Test", control_inputs=None, control_labels=None):
        """
        Modular function to plot trajectory for stability testing.
        Args:
            trajectory: Tensor of shape (time_steps, n_states) or (time_steps, batch_size, n_states)
            dt: Time step for x-axis
            state_labels: List of labels for each state (e.g., ['x1', 'x2'])
            title: Plot title
            control_inputs: Optional tensor of shape (time_steps-1, n_controls) or (time_steps-1, batch_size, n_controls)
            control_labels: List of labels for control inputs
        """
        
        # Generalize to handle any tensor dimension by flattening extra dimensions
        # Assume first dim is time_steps, flatten the rest into features
        if trajectory.dim() > 2:
            trajectory = trajectory.reshape(trajectory.shape[0], -1)
        elif trajectory.dim() == 1:
            # If 1D, treat as single feature over time
            trajectory = trajectory.unsqueeze(-1)
        # Now trajectory is 2D: (time_steps, n_features)
            
        trajectory = trajectory.cpu().numpy()
        time_steps = np.arange(len(trajectory)) * dt
        n_states = trajectory.shape[1]
        
        # Default labels if not provided
        if state_labels is None:
            state_labels = [f"Feature {i+1}" for i in range(n_states)]
        
        plt.figure(figsize=(10, 6))
        
        # Plot states
        colors = plt.cm.tab10.colors  # Use tab10 colormap for distinct colors
        for i in range(n_states):
            plt.plot(time_steps, trajectory[:, i], label=state_labels[i], color=colors[i % len(colors)])
        
        # Plot control inputs if provided
        if control_inputs is not None:
            # Generalize: flatten extra dimensions
            if control_inputs.dim() > 2:
                control_inputs = control_inputs.reshape(control_inputs.shape[0], -1)
            elif control_inputs.dim() == 1:
                control_inputs = control_inputs.unsqueeze(-1)
            
            n_controls = control_inputs.shape[1]
            control_inputs = control_inputs.cpu().numpy()
            
            if control_labels is None:
                control_labels = [f"Control {i+1}" for i in range(n_controls)]
            elif len(control_labels) != n_controls:
                raise ValueError(f"control_labels must have {n_controls} elements")
            
            for i in range(n_controls):
                plt.plot(time_steps[:-1], control_inputs[:, i], label=control_labels[i], linestyle='--', 
                        color=colors[(n_states + i) % len(colors)])
        
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()

        print("Final State:", trajectory[-1, :])
        
    def Q_R_matrices(self):
        if self.Q is None:
            self.Q = np.eye(self.in_dim)
            print("Q matrix not defined, using identity matrix.")
        if self.R is None:
            self.R = np.eye(self.out_dim)
            print("R matrix not defined, using identity matrix.")
        return self.Q, self.R 
        
    
        


