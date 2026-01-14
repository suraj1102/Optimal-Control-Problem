from models.valuefunctionmodel import ValueFunctionModel
from tqdm import tqdm
import torch

class Pinn(ValueFunctionModel):
    def __init__(self, hparams):
        super().__init__(hparams)

    
    def get_outputs(self, x: torch.Tensor):
        x.requires_grad_(True)

        g_x = self(x)
        g_0 = self(self.x_bc)
        v = g_x

        grad_v = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]

        return g_x, g_0, v, grad_v
    

    def train(self):
        self.set_optimizer_scheduler()
        super().train()

        progress_bar = tqdm(range(self.hparams.training_params.n_epochs), desc="Training Progress", unit="epoch")
        for epoch in progress_bar:
            self.optimizer.zero_grad()

            x_colloc = self.sample_inputs()
            x_colloc.requires_grad_(True)

            g_x, g_0, v, grad_v = self.get_outputs(x_colloc)
            pde_residual = self.problem.pde_residual(x_colloc, grad_v)
            boundary_residual = self.v_bc - g_0

            boundary_loss = torch.mean(boundary_residual**2)
            pde_loss = torch.mean(pde_residual**2)

            loss = pde_loss + boundary_loss
            loss.backward()

            self.optimizer.step()

            progress_bar.set_postfix({
                "Total Loss": loss.item(),
                "PDE Loss": pde_loss.item(),
                "Boundary Loss": boundary_loss.item()
            })