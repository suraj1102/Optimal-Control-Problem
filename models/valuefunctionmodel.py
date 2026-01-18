import torch
import torch.nn as nn
import torch.optim as optim
from models.hparams import Hyperparams
from models.problem import problem
import numpy as np

class ValueFunctionModel(torch.nn.Module):
    def __init__(self, problem: problem):
        super().__init__()
        self.hparams = problem.hparams
        self.problem = problem

        hidden_units = self.hparams.training_params.hidden_units
        in_dim = self.hparams.problem_params.in_dim
        out_dim = 1

        self.device = self.hparams.device.device

        # Use ModuleList so submodules are registered and moved when calling .to(device)
        self.layers: nn.ModuleList = nn.ModuleList()

        for i, _ in enumerate(hidden_units):
            hidden = nn.Linear((in_dim if i == 0 else hidden_units[i - 1]), hidden_units[i])
            self.layers.append(hidden)

        for i, layer in enumerate(self.layers):
            assert isinstance(layer, nn.Linear) # To get rid of LSP error 
            nn.init.uniform_(layer.weight, -1, 1)
            nn.init.uniform_(layer.bias, -1, 1)

        self.y = nn.Linear(hidden_units[-1], out_dim, bias=self.hparams.hyper_params.bias)
        nn.init.uniform_(self.y.weight, -1, 1)

        self.activation = self.hparams.training_params.activation()

        self.x_bc = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=self.device)
        self.v_bc = torch.tensor([[0.0]], dtype=torch.float32, device=self.device)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.y(self.hidden_layers_output(x))
    
    def hidden_layers_output(self, x: torch.Tensor) -> torch.Tensor:
        for _, layer in enumerate(self.layers):
            x = self.activation(layer(x))
        return x
    
    def set_optimizer_scheduler(self):
        optimizer_name = self.hparams.optimizer_params.optimizer
        lr = self.hparams.optimizer_params.lr

        if optimizer_name in ['ADAM', 'Adam', 'adam', torch.optim.Adam, optim.Adam]:
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Optimizer {optimizer_name} not recognized or implemented.")
        
    def sample_inputs(self, num_points: int = None) -> torch.Tensor:
        xs = []

        n_sample = self.hparams.training_params.n_colloc if num_points is None else num_points
        input_ranges = self.hparams.problem_params.input_ranges
        edge_weights = self.hparams.training_params.edge_sampling_weight

        for i in range(self.hparams.problem_params.in_dim):
            n_edge = int(n_sample * edge_weights[i] / 2)
            n_mid = int((1 - edge_weights[i]) * n_sample)
            span = abs(input_ranges[i][1] - input_ranges[i][0])

            edge_low = np.random.uniform(input_ranges[i][0], input_ranges[i][0] + edge_weights[i] * span, size=(n_edge, 1))
            edge_high = np.random.uniform(input_ranges[i][1] - edge_weights[i] * span, input_ranges[i][1], size=(n_edge, 1))
            mid = np.random.uniform(input_ranges[i][0] + edge_weights[i] * span, input_ranges[i][1] - edge_weights[i] * span, size=(n_mid, 1))

            xi = np.vstack((edge_low, mid, edge_high))
            xs.append(xi)

        x = np.hstack(xs)
        return torch.tensor(x, dtype=torch.float32, device=self.device)
    
    def _generate_trajectory(self, x0: torch.Tensor, step_size: float, n_steps: int) -> torch.Tensor:
        trajectory = [x0]

        x_current = x0
        for _ in range(n_steps):
            x_current.requires_grad_(True)
            _, _, _, grad_v = self.get_outputs(x_current)

            f_x = self.problem.f_x(x_current)
            g_x = self.problem.g_x(x_current)

            u_star = self.problem.control_input(x_current, grad_v)

            x_dot = f_x + g_x * u_star
            x_next = x_current + step_size * x_dot

            trajectory.append(x_next)
            x_current = x_next

        return torch.cat(trajectory, dim=0)
    

    def plot_trajectory(self, x0: torch.Tensor, step_size: float, n_steps: int):
        import matplotlib.pyplot as plt

        trajectory = self._generate_trajectory(x0, step_size, n_steps).detach().cpu().numpy()

        plt.figure(figsize=(8, 6))
        time = np.arange(trajectory.shape[0])
        plt.plot(time, trajectory[:, 0], label='x1')
        plt.plot(time, trajectory[:, 1], label='x2')
        plt.title('Generated Trajectory')
        plt.xlabel('Time step')
        plt.ylabel('State value')
        plt.legend()
        plt.grid()
        plt.show()


    def plot_value_function(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        input_ranges = self.hparams.problem_params.input_ranges
        n_points = 100

        x1 = np.linspace(input_ranges[0][0], input_ranges[0][1], n_points)
        x2 = np.linspace(input_ranges[1][0], input_ranges[1][1], n_points)
        X1, X2 = np.meshgrid(x1, x2)
        inputs = np.stack([X1.ravel(), X2.ravel()], axis=1)
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            values = self.forward(inputs_tensor).cpu().numpy().reshape(X1.shape)
            if values.shape != X1.shape:
                # If output is (N, 1), squeeze to (N,)
                values = values.squeeze()
                values = values.reshape(X1.shape)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X1, X2, values, cmap='viridis')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('Value')
        ax.set_title('Value Function Surface')
        plt.show()