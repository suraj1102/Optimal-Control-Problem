from architectures.xtfc import XTFC
import torch
from tqdm import tqdm

class XTFC_Unfreeze(XTFC):
    def __init__(self, problem):
        super().__init__(problem)

    def train_(self):
        self.set_optimizer_scheduler()
        self.train()


        progress_bar = tqdm(range(self.hparams.training_params.n_epochs), desc="Training Progress", unit="epoch")
        for epoch in progress_bar:
            if self.hparams.hyper_params.training_visualization and epoch % 1000 == 0:
                self.plot_value_function()

            self.optimizer.zero_grad()

            x_colloc = self.sample_inputs()
            x_colloc.requires_grad_(True)

            g_x, g_0, v, grad_v = self.get_outputs(x_colloc)
            pde_residual = self.problem.pde_residual(x_colloc, grad_v)
            boundary_residual = self.v_bc - g_0

            boundary_loss = torch.mean(boundary_residual**2) # Just for display
            pde_loss = torch.mean(pde_residual**2)

            # l1_loss = self.hparams.training_params.l1_lambda * sum(param.abs().sum() for param in self.parameters())
            # l2_loss = self.hparams.training_params.l2_lambda * sum((param ** 2).sum() for param in self.parameters())
            l1_loss, l2_loss = 0, 0

            total_loss = pde_loss + l1_loss + l2_loss

            total_loss.backward()

            self.optimizer.step()

            progress_bar.set_postfix({
                "PDE Loss": pde_loss.item(),
                "Boundary Loss": boundary_loss.item(),
                "Total Loss": total_loss.item()
            })