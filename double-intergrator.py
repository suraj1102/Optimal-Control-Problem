from problem import *

class double_integrator(problem):
    def __init__(self):
        super().__init__()
        self.problem = "double-integrator"
        self.architecture = "xtfc"
        self.analytical_pretraining = "xTQx"
        self.in_dim = 2
        self.out_dim = 1
        self.hidden_units = [64, 64, 64]
        self.activation = nn.Tanh()
        self.n_colloc = 1000
        self.input_range = [-1.0, 1.0]
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
        self.model_save_path = "models/double_integrator_model.pth"
        self.save_plot = True
        self.initial_conditions = [-0.5, 0.5]

        self.Q = np.diag([1.0, 1.0])
        self.R = np.diag([0.1])
        self.LOAD_MODEL = False

        # Override parent functions with custom implementations
        self.compute_pde_residual = self.pde_residual
        self.compute_control_input = self.control_input
        self.f_x = self.f_x_di
        self.g_x = self.g_x_di

    def f_x_di(self, x: torch.Tensor) -> torch.Tensor:
        # Drift term for double integrator: \dot{x} = [x2, 0] + [0, 1] * u
        return torch.stack([
            x[:, 1],  # \dot{x1} = x2
            torch.zeros_like(x[:, 0], device=x.device)  # \dot{x2} = 0 (drift)
        ], dim=1)

    def g_x_di(self, x: torch.Tensor) -> torch.Tensor:
        # Control matrix: \dot{x} = f_x + g_x * u
        return torch.stack([
            torch.zeros_like(x[:, 0], device=x.device),  # u affects x2
            torch.ones_like(x[:, 0], device=x.device)    # u = \dot{x2}
        ], dim=1)

    def control_input(self, x: torch.Tensor, grad_v: torch.Tensor) -> torch.Tensor:
        # Ensure Q and R are defined, using defaults if not
        Q, R = self.Q_R_matrices()
        
        # LQR control: u = -0.5 * R^{-1} * g_x^T * ∇V
        Q = torch.tensor(Q, device=x.device, dtype=torch.float32)
        R = torch.tensor(R, device=x.device, dtype=torch.float32)
        
        # For double integrator, g_x = [0, 1], so g_x^T * ∇V = [0, 1] * [V_x1, V_x2] = V_x2
        # u = -0.5 * R * V_x2 (since R is scalar)
        return -0.5 * R[0, 0] * grad_v[:, 1:2]  # grad_v[:, 1] is V_x2

    def pde_residual(self, x: torch.Tensor, grad_v: torch.Tensor) -> torch.Tensor:
        # HJB equation residual for double integrator
        x1 = x[:, 0]
        x2 = x[:, 1]
        V_x1 = grad_v[:, 0]
        V_x2 = grad_v[:, 1]
        
        # Residual: -0.5*(x1^2 + x2^2) - V_x1*x2 + 0.5*V_x2^2
        term1 = -0.5 * (torch.square(x1) + torch.square(x2))
        term2 = -V_x1 * x2
        term3 = 0.5 * torch.square(V_x2)
        return term1 + term2 + term3