from problem import *

class double_input_control_problem(problem):
    def __init__(self):
        super().__init__()
        self.problem = "double-input-control-problem"
        self.architecture = "xtfc"
        self.analytical_pretraining = "xTQx"
        self.in_dim = 2
        self.out_dim = 2
        self.hidden_units = [64, 64, 64]
        self.activation = nn.Tanh()
        self.n_colloc = 1000
        self.input_range = [(-1.0, 1.0), (-1.0, 1.0)]
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
        self.model_save_path = "models/double_input_control_problem_model.pth"
        self.save_plot = True

        self.Q = np.diag([1.0, 1.0])
        self.R = np.diag([1.0, 1.0])
        self.LOAD_MODEL = False

        self.initial_conditions = [0.1, 0.1]

        # Override parent functions with custom implementations
        self.compute_pde_residual = self.pde_residual
        self.compute_control_input = self.control_input
        self.f_x = self.f_x_dicp
        self.g_x = self.g_x_dicp

    def test_stability(self, trajectory, dt=0.01, title="Double Input Control Problem Stability Test", control_inputs=None):
        state_labels = ['x1', 'x2']
        control_labels = ['u1', 'u2']
        super().test_stability(trajectory, dt, state_labels, title, control_inputs, control_labels)

    def f_x_dicp(self, x: torch.Tensor) -> torch.Tensor:
        # Drift term for double input control problem: \dot{x} = [0, 0] + g_x * u
        return torch.zeros_like(x)

    def g_x_dicp(self, x: torch.Tensor) -> torch.Tensor:
        # Control matrix: \dot{x} = f_x + g_x @ u
        # g_x is identity matrix for each sample
        return torch.eye(2, device=x.device).unsqueeze(0).expand(x.shape[0], -1, -1)

    def control_input(self, x: torch.Tensor, grad_v: torch.Tensor) -> torch.Tensor:
        # Ensure Q and R are defined, using defaults if not
        Q, R = self.Q_R_matrices()
        
        Q = torch.tensor(Q, device=x.device, dtype=torch.float32)
        R = torch.tensor(R, device=x.device, dtype=torch.float32)
        
        # LQR control: u = - R^{-1} @ grad_v.T (since g_x = I, g_x^T = I)
        # Solve R @ u.T = - grad_v.T, so u.T = - R^{-1} @ grad_v.T
        u_T = -torch.linalg.solve(R, grad_v.T)
        return u_T.T  # Shape: (batch, out_dim)

    def pde_residual(self, x: torch.Tensor, grad_v: torch.Tensor) -> torch.Tensor:
        # HJB equation residual for double input control problem
        x1 = x[:, 0]
        x2 = x[:, 1]
        V_x1 = grad_v[:, 0]
        V_x2 = grad_v[:, 1]
        
        # Residual: -0.5*(x1^2 + x2^2) + 0.5*(V_x1^2 + V_x2^2)
        term1 = -0.5 * (torch.square(x1) + torch.square(x2))
        term2 = 0.5 * (torch.square(V_x1) + torch.square(V_x2))
        return term1 + term2