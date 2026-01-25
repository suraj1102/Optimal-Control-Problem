import torch
import torch.nn as nn
import torch.optim as optim
from models.hparams import Hyperparams
from models.problem import problem
import numpy as np
import matplotlib.pyplot as plt
import logging

class ValueFunctionModel(torch.nn.Module):
    def __init__(self, problem: problem):
        super().__init__()
        self.hparams = problem.hparams
        self.problem = problem

        self.logger = self.hparams.logger

        hidden_units = self.hparams.training_params.hidden_units
        in_dim = self.hparams.problem_params.in_dim
        out_dim = 1

        self.device = self.hparams.device.device
        self.debug = self.hparams.hyper_params.debug

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

        if self.debug:
            self.logger.debug(f"Model initialized on device: {self.device}")
            self.logger.debug(f"Input dimension: {in_dim}, Output dimension: {out_dim}")
            self.logger.debug(f"Hidden units: {hidden_units}")
            self.logger.debug(f"Layers:")
            for i, layer in enumerate(self.layers):
                self.logger.debug(f"  Layer {i}: {layer}")
            self.logger.debug(f"Output layer: {self.y}")
            self.logger.debug(f"Activation function: {self.activation}")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.y(self.hidden_layers_output(x))
    
    def hidden_layers_output(self, x: torch.Tensor) -> torch.Tensor:
        for _, layer in enumerate(self.layers):
            x = self.activation(layer(x))
        return x
    
    def get_outputs(self, x: torch.Tensor):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def set_optimizer_scheduler(self):
        optimizer_name = self.hparams.optimizer_params.optimizer
        lr = self.hparams.optimizer_params.lr

        if optimizer_name in ['ADAM', 'Adam', 'adam', torch.optim.Adam, optim.Adam]:
            self.optimizer = optim.Adam(self.parameters(), lr=lr)

            self.logger.debug(f"Using Adam optimizer with learning rate {lr}")
        else:
            raise ValueError(f"Optimizer {optimizer_name} not recognized or implemented.")
        
    def sample_inputs(self, num_points: int = None) -> torch.Tensor:
        n_sample = self.hparams.training_params.n_colloc if num_points is None else num_points

        input_ranges = self.hparams.problem_params.input_ranges
        in_dim = self.hparams.problem_params.in_dim

        x = np.zeros((n_sample, in_dim))

        for d in range(in_dim):
            low, high = input_ranges[d]
            x[:, d] = np.random.uniform(low, high, size=n_sample)

        return torch.tensor(x, dtype=torch.float32, device=self.device)
    
    def plot_sample_inputs(self, x: torch.Tensor):
        x = x.cpu().detach().numpy()
        if x.shape[1] == 2:
            plt.scatter(x[:, 0], x[:, 1], alpha=0.6)
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.title('Sampled Inputs')
            plt.grid(True)
            plt.show()
        else:
            plt.plot(x)
            plt.title('Sampled Inputs')
            plt.grid(True)
            plt.show()
    
    def _generate_trajectory(self, x0: torch.Tensor, step_size: float, time: int) -> torch.Tensor:
        trajectory = [x0]
        u = []

        x_current = x0
        n_steps = int(time / step_size)
        for step in range(n_steps):
            x_current.requires_grad_(True)
            _, _, v, grad_v = self.get_outputs(x_current)

            f_x = self.problem.f_x(x_current)
            g_x = self.problem.g_x(x_current)

            u_star = self.problem.control_input(x_current, grad_v)


            x_dot = f_x + g_x * u_star
            x_next = x_current + step_size * x_dot

            # if self.debug:
            #     self.logger.info(f"timestep: {step}")
            #     self.logger.info(f"x_current: {x_current.data}")
            #     self.logger.info(f"V(x_current) = {v.data}")
            #     self.logger.info(f"del_V/del_x = {grad_v.data}")
            #     self.logger.info(f"u_star: {u_star.data}")

            # ---- FOR IP -----
            if self.hparams.hyper_params.problem.lower() == "inverted-pendulum":
                x_next = (x_next + torch.pi) % (2 * torch.pi) - torch.pi

            trajectory.append(x_next)
            u.append(u_star)
            x_current = x_next



        return torch.cat(trajectory, dim=0), torch.cat(u, dim=0)
    

    def plot_trajectory(self, x0: torch.Tensor, step_size: float, time: int):
        n_steps = int(time / step_size)
        trajectory, u = self._generate_trajectory(x0, step_size, n_steps)

        trajectory = trajectory.cpu().detach().numpy()
        trajectory = trajectory[1:, :] # Remove first entry as that is the initial condition
        u = u.cpu().detach().numpy()

        labels = self.hparams.problem_params.labels

        self.logger.info(f"Full trajectory shape = {trajectory.shape}")
        self.logger.info(f"Full u shape = {u.shape}")

        plt.figure(figsize=(8, 6))
        time = np.arange(trajectory.shape[0])

        for i in range(trajectory.shape[1]):
            plt.plot(time, trajectory[:, i], label=labels[i] if labels and i < len(labels) else f'x{i+1}')

        plt.plot(time, u, label='Control', color='blue')

        plt.title('Generated Trajectory')
        plt.xlabel('Time step')
        plt.ylabel('State value')
        plt.legend()
        plt.grid()
        plt.show()


    def plot_value_function(self):
        input_ranges = self.hparams.problem_params.input_ranges
        n_points = 100

        x1 = np.linspace(input_ranges[0][0], input_ranges[0][1], n_points)
        x2 = np.linspace(input_ranges[1][0], input_ranges[1][1], n_points)
        X1, X2 = np.meshgrid(x1, x2)
        inputs = np.stack([X1.ravel(), X2.ravel()], axis=1)
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        
        g_x, _, _, _ = self.get_outputs(inputs_tensor)

        values = g_x.cpu().detach().numpy().reshape(X1.shape)

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