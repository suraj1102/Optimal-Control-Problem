import torch
import torch.nn as nn
import torch.optim as optim
from models.hparams import Hyperparams
from models.problem import problem
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

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
    
    
    def pre_train_step(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def train_step(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def train_model(self):
        self.train() # Set model to training mode (as opposed to eval)
        self.set_optimizer_scheduler()

        self.pre_train_step()

        progress_bar = tqdm(range(self.hparams.training_params.n_epochs), desc="Training Progress")
        for _ in progress_bar:

            loss = self.train_step()

            if self.hparams.training_params.l1_lambda != 0:
                l1_norm = sum(p.abs().sum() for p in self.parameters())
                loss += self.hparams.training_params.l1_lambda * l1_norm

            if self.hparams.training_params.l2_lambda != 0:
                l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
                loss += self.hparams.training_params.l2_lambda * l2_norm

            loss.backward()
            self.optimizer.step()

            progress_bar.set_postfix({"Loss": loss.item()})