from problem import problem
import numpy as np
import torch
import torch.nn as nn

class inverted_pendulum(problem):   
    def __init__(self):
        super().__init__()
        self.problem = "inverted-pendulum"
        self.architecture = "xtfc"
        self.analytical_pretraining = "xTQx"
        self.in_dim = 2
        self.out_dim = 1
        self.hidden_units = [64, 64, 64]
        self.activation = nn.Tanh()
        self.n_colloc = 1000
        self.input_range = [-np.pi / 6, np.pi / 6]
        self.edge_sampling_weight = 0.0
        self.lr = 1e-3
        self.optimizer = "ADAM"
        self.Scheduler = "ReduceLROnPlateau"
        self.patience = 20
        self.gamma = 0.9
        self.n_epochs = 5000
        self.early_stopping = 100
        self.log_wandb = True
        self.plot_graphs = True
        self.save_model = True
        self.model_save_path = "models/inverted_pendulum_model.pth"
        self.save_plot = True

        self.Q = np.diag([1.0, 1.0])
        self.R = np.diag([0.1])
        self.LOAD_MODEL = False

        # Physical parameters for inverted pendulum
        self.mass = 0.2
        self.length = 1.0
        self.gravity = 9.81
        self.initial_conditions = [0.1, 0.0]

        # Override parent functions with custom implementations
        self.compute_pde_residual = self.pde_residual
        self.compute_control_input = self.control_input
        self.f_x = self.f_x_ip
        self.g_x = self.g_x_ip

    def test_stability(self, trajectory, dt=0.01, title="Inverted Pendulum Stability Test", control_inputs=None):
        state_labels = ['theta (Angle)', 'thetadot (Angular Velocity)']
        control_labels = ['u (Torque)']
        super().test_stability(trajectory, dt, state_labels, title, control_inputs, control_labels)

    def f_x_ip(self, x: torch.Tensor) -> torch.Tensor:
        # Drift term for inverted pendulum: \dot{x} = [thetadot, (g/l) * sin(theta)]
        return torch.stack([
            x[:, 1],  # \dot{theta} = thetadot
            self.gravity / self.length * torch.sin(x[:, 0])  # \dot{thetadot} = (g/l) * sin(theta)
        ], dim=1)

    def g_x_ip(self, x: torch.Tensor) -> torch.Tensor:
        # Control matrix: \dot{x} = f_x + g_x * u
        return torch.stack([
            torch.zeros_like(x[:, 0], device=x.device),  # u affects thetadot
            torch.ones_like(x[:, 0], device=x.device) / (self.mass * self.length * self.length)  # u = \dot{thetadot}
        ], dim=1)

    def control_input(self, x: torch.Tensor, grad_v: torch.Tensor) -> torch.Tensor:
        # Ensure Q and R are defined, using defaults if not
        Q, R = self.Q_R_matrices()
        
        Q = torch.tensor(Q, device=x.device, dtype=torch.float32)
        R = torch.tensor(R, device=x.device, dtype=torch.float32)
        
        # For inverted pendulum, g_x = [0, 1/(m*l^2)], so g_x^T * ∇V = V_thetadot / (m*l^2)
        # u = -0.5 * R^{-1} * g_x^T * ∇V
        return -0.5 * R[0, 0] * grad_v[:, 1:2] / (self.mass * self.length * self.length)  # grad_v[:, 1] is V_thetadot

    def pde_residual(self, x: torch.Tensor, grad_v: torch.Tensor) -> torch.Tensor:
        # HJB equation residual for inverted pendulum
        theta = x[:, 0]
        thetadot = x[:, 1]
        V_theta = grad_v[:, 0]
        V_thetadot = grad_v[:, 1]
        
        # Residual: -0.5*(theta^2 + thetadot^2) - V_theta*thetadot + 0.5*(V_thetadot^2)/(m*l^2) + V_theta * (gravity/l * sin(theta))
        term1 = -0.5 * (torch.square(theta) + torch.square(thetadot))
        term2 = -V_theta * thetadot
        term3 = 0.5 * torch.square(V_thetadot) / (self.mass * self.length * self.length)
        term4 = V_theta * (self.gravity / self.length * torch.sin(theta))
        return term1 + term2 + term3 + term4
