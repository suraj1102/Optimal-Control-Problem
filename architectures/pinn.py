from models.valuefunctionmodel import ValueFunctionModel
from models.problem import problem
from tqdm import tqdm
import torch

class Pinn(ValueFunctionModel):
    def __init__(self, problem: problem):
        super().__init__(problem)

    
    def get_outputs(self, x: torch.Tensor):
        x.requires_grad_(True)

        g_x = self(x)
        g_0 = self(self.x_bc)
        v = g_x

        grad_v = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]

        return g_x, g_0, v, grad_v
    

    def pre_train_step(self):
        pass

    def train_step(self):
        self.optimizer.zero_grad()

        x_colloc = self.sample_inputs()
        x_colloc.requires_grad_(True)

        _, g_0, _, grad_v = self.get_outputs(x_colloc)
        pde_residual = self.problem.pde_residual(x_colloc, grad_v)
        boundary_residual = self.v_bc - g_0

        boundary_loss = torch.mean(boundary_residual**2)
        pde_loss = torch.mean(pde_residual**2)

        loss = pde_loss + boundary_loss
        return loss