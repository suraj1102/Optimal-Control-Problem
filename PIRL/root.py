#imports
import torch
import torch.nn as nn
import numpy as np

class Actor(nn. Module):

    def __init__(self, state_dim, action_dim, action_bound):
        super().__init__()

        self.bound = action_bound
        self.net = nn.Sequential(
        nn.Linear(state_dim, 64, bias=False),
        nn.Tanh(),
        nn.Linear(128, 128, bias=False),
        nn.Tanh(),
        nn.Linear(128, action_dim, bias=False),
        nn.Tanh()    
        )

    def forward(self, x):
            return self.net(x)*self.bound
    

class Critic(nn.Module):
    def __init__(self, state_dim):
          super().__init__()
          self.features = nn.sequential(
               nn.Linear(state_dim, 128, bias = False),
               nn.Tanh()
               nn.Linear(128, state_dim, bias = False)
          )
     
    def forward(self, x):
        phi = self.features(x)
        return torch.sum(phi * phi, dim=-1, keepdim=True)

class AdmissibleNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, bound):
        super().__init__(in_dim, out_dim, hidden_dim)
        self.bound = bound

    def forward(self, x):
        return torch.tanh(self.net(x)) * self.bound
    



    