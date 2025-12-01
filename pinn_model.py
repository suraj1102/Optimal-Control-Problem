from torch import nn
import torch
from hparams import hparams

class Pinn(nn.Module):
    def __init__(self, in_dim = 2, out_dim=1, hparams=hparams):
        super(Pinn, self).__init__()
        
        self.layers = nn.ModuleList()
        self.act = torch.tanh

        for i, num_hidden_units in enumerate(hparams['hidden_units']):
            hi = nn.Linear((in_dim if i == 0 else hparams['hidden_units'][i - 1]), hparams['hidden_units'][i])
            self.layers.append(hi)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return x
    
    def get_outputs(self, x: torch.Tensor):
        x.requires_grad_(True)

        g_x = self(x)  # N, 1
        g_0 = self(self.x_bc)

        v = g_x + self.v_bc - g_0

        grad_v = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        # grad_v is N, 2 | 2 is in_dim

        return g_x, g_0, v, grad_v