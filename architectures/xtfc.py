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

    
    def xTQx_analytical_pretraining(self):
        x = self.sample_inputs(self.hparams.pretraining_params.n_pretraining_colloc)
        Q = self.hparams.problem_params.Q

        with torch.no_grad():
            H = self.hidden_layers_output(x)
            target = x.T @ Q @ x

            # Analytical solution: W = (HᵀH + λI)⁻¹ Hᵀ T
            HTH = H.T @ H
            HTT = H.T @ target

            I = torch.eye(HTH.shape[0], device=HTH.device)
            HTH_reg = HTH + self.hparams.pretraining_params.lambda_reg * I

            beta_analytical = torch.linalg.solve(HTH_reg, HTT)

            self.y.weight.data = beta_analytical.T

            V_approx = H @ beta_analytical
            mse_error = torch.mean((V_approx - target) ** 2).item()

            print(f"Pretraining completed. MSE Error: {mse_error}")