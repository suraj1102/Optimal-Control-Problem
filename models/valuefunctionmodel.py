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
        return self.y(self.forward_layers_output(x))
    
    def forward_layers_output(self, x: torch.Tensor) -> torch.Tensor:
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
        
    def sample_inputs(self) -> torch.Tensor:
        xs = []

        n_sample = self.hparams.training_params.n_colloc
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