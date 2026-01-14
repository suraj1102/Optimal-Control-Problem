import torch
import torch.nn as nn
from models.hparams import Hyperparams

class ValueFunctionModel(torch.nn.Module):
    def __init__(self, hparams: Hyperparams):
        super().__init__()
        self.hparams = hparams

        hidden_units = hparams.training_params.hidden_units
        in_dim = hparams.problem_params.in_dim
        out_dim = hparams.problem_params.out_dim

        self.device = hparams.training_params.device

        # Use ModuleList so submodules are registered and moved when calling .to(device)
        self.layers: nn.ModuleList = nn.ModuleList()

        for i, _ in enumerate(hidden_units):
            hidden = nn.Linear((in_dim if i == 0 else hidden_units[i - 1]), hidden_units[i])
            self.layers.append(hidden)

        for i, layer in enumerate(self.layers):
            assert isinstance(layer, nn.Linear) # To get rid of LSP error 
            nn.init.uniform_(layer.weight, -1, 1)
            nn.init.uniform_(layer.bias, -1, 1)

        self.y = nn.Linear(hidden_units[-1], out_dim, bias=hparams.hyper_params.bias)
        nn.init.uniform_(self.y.weight, -1, 1)

        self.activation = hparams.training_params.activation()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.y(self.forward_layers_output(x))
    
    def forward_layers_output(self, x: torch.Tensor) -> torch.Tensor:
        for _, layer in enumerate(self.layers):
            x = self.activation(layer(x))
        return x
    
