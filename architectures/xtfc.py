from models.valuefunctionmodel import ValueFunctionModel
from models.problem import problem
from tqdm import tqdm
import torch

class XTFC(ValueFunctionModel):
    def __init__(self, problem: problem):
        super().__init__(problem)

    def get_outputs(self, x: torch.Tensor):
        x.requires_grad_(True)

        g_x = self(x)
        g_0 = self(self.x_bc)
        v = g_x + self.v_bc - g_0
        grad_v = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]


        return g_x, g_0, v, grad_v
    
    def freeze_hidden(self):
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = False

        for param in self.y.parameters():
            param.requires_grad = True

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def _get_target(self, x: torch.Tensor):
        Q = self.hparams.problem_params.Q

        # Q should be square (n, n), x should be (batch_size, n)
        # targets should be (batch_size, 1)
        # Compute x^T Q x for each sample in the batch
        targets = torch.sum(x @ Q * x, dim=1, keepdim=True)

        return targets
    
    def analytical_pretraining(self):
        if self.hparams.hyper_params.analytical_pretraining.lower() == "xtqx":
            tries = 1 if self.hparams.pretraining_params.initialization_cutoff == -1 else self.hparams.pretraining_params.init_limit
            best_mse = float('inf')
            best_beta = None

            for attempt in range(tries):
                mse_error, beta_analytical = self.perform_xTQX()

                if mse_error < self.hparams.pretraining_params.initialization_cutoff:
                    print(f"Pretraining successful on attempt {attempt + 1} with MSE Error: {mse_error}")
                    break
                
                print(f"Pretraining attempt {attempt + 1} failed with MSE Error: {mse_error}. Retrying...")
                if mse_error < best_mse:
                    best_mse = mse_error
                    best_beta = beta_analytical
            else:
                print(f"Pretraining did not meet cutoff after {tries} attempts. Using best result with MSE Error: {best_mse}")
                self.y.weight.data = best_beta.T

    def perform_xTQX(self):
        x = self.sample_inputs(self.hparams.pretraining_params.n_pretraining_colloc)
        Q = self.hparams.problem_params.Q

        with torch.no_grad():
            H = self.hidden_layers_output(x)
            target = self._get_target(x)

            # Analytical solution: W = (HᵀH + λI)⁻¹ Hᵀ T
            HTH = H.T @ H
            HTT = H.T @ target

            I = torch.eye(HTH.shape[0], device=HTH.device)
            HTH_reg = HTH + self.hparams.pretraining_params.lambda_reg * I

            beta_analytical = torch.linalg.solve(HTH_reg, HTT)

            self.y.weight.data = beta_analytical.T

            V_approx = H @ beta_analytical
            mse_error = torch.mean((V_approx - target) ** 2).item()

            if self.debug:
                print(f"x shape: {x.shape}")
                print(f"Q shape: {Q.shape}")
                print(f"H shape: {H.shape}")
                print(f"target shape: {target.shape}")
                print(f"HTH shape: {HTH.shape}")
                print(f"HTT shape: {HTT.shape}")
                print(f"I shape: {I.shape}")
                print(f"HTH_reg shape: {HTH_reg.shape}")
                print(f"beta_analytical shape: {beta_analytical.shape}")
                print(f"self.y.weight shape: {self.y.weight.data.shape}")
                print(f"V_approx shape: {V_approx.shape}")

                self.plot_sample_inputs(x)

            print(f"Pretraining completed. MSE Error: {mse_error}")

            return mse_error, beta_analytical

    def train_(self):
        self.train() # Set model to training mode (as opposed to eval)

        self.freeze_hidden()

        self.set_optimizer_scheduler()

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

            pde_loss.backward()

            self.optimizer.step()

            progress_bar.set_postfix({
                "PDE Loss": pde_loss.item(),
                "Boundary Loss": boundary_loss.item()
            })