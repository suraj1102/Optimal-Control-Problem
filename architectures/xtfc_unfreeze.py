from architectures.xtfc import XTFC
import torch
from tqdm import tqdm

class XTFC_Unfreeze(XTFC):
    def __init__(self, problem):
        super().__init__(problem)

    def train_(self):
        self.set_optimizer_scheduler()
        self.train()

        print("x_bc requires_grad:", self.x_bc.requires_grad)
        print("x_bc grad_fn:", self.x_bc.grad_fn)


        progress_bar = tqdm(range(self.hparams.training_params.n_epochs), desc="Training Progress", unit="epoch")
        for _ in progress_bar:
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