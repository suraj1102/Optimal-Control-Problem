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

    def _get_xtqx_target(self, x: torch.Tensor):
        Q = self.hparams.problem_params.Q

        # Q should be square (n, n), x should be (batch_size, n)
        # targets should be (batch_size, 1)
        # Compute x^T Q x for each sample in the batch
        targets = torch.sum(x @ Q * x, dim=1, keepdim=True)

        return targets
    

    def _linearize_and_solve_care(self, x_eq=None, u_eq=None):
        """
        Automatically linearizes the system using autograd and
        solves the continuous-time CARE.

        CARE - continuous-time algebraic riccati equation

        Returns:
            A, B, S, K
        """
        import numpy as np
        from scipy.linalg import solve_continuous_are

        device = self.device
        
        # Cost matrices (from your hparams)
        Q = self.hparams.problem_params.Q.detach().cpu().numpy()
        R = self.hparams.problem_params.R.detach().cpu().numpy()

        # Equilibrium point
        if x_eq is None:
            x_eq = torch.zeros(1, Q.shape[0], device=device, requires_grad=True)
        else:
            x_eq = x_eq.clone().detach().requires_grad_(True)

        if u_eq is None:
            u_eq = torch.zeros(1, R.shape[0], device=device, requires_grad=True)
        else:
            u_eq = u_eq.clone().detach().requires_grad_(True)

        # Define full dynamics
        def dynamics(x, u):
            return self.problem.f_x(x) + self.problem.g_x(x) * u

        # ---- Compute A = df/dx ----
        A = torch.autograd.functional.jacobian(
            lambda x: dynamics(x, u_eq),
            x_eq
        ).squeeze()

        # ---- Compute B = df/du ----
        B = torch.autograd.functional.jacobian(
            lambda u: dynamics(x_eq, u),
            u_eq
        ).squeeze()

        # Convert to numpy
        A_np = A.detach().cpu().numpy()
        B_np = B.detach().cpu().numpy().reshape(2, 1)


        # ---- Solve CARE ----
        S = solve_continuous_are(A_np, B_np, Q, R)

        # LQR gain (optional but usually useful)
        K = np.linalg.inv(R) @ B_np.T @ S

        return A_np, B_np, S, K

    
    def analytical_pretraining(self):
        if self.hparams.hyper_params.analytical_pretraining.lower() == "xtqx":
            pretraning_function = self.perform_xTQX

        if self.hparams.hyper_params.analytical_pretraining.lower() == "lqr":
            pretraning_function = self.perform_LQR_pretraining

        tries = 1 if self.hparams.pretraining_params.initialization_cutoff == -1 else self.hparams.pretraining_params.init_limit
        best_mse = float('inf')
        best_beta = None

        for attempt in range(tries):
            mse_error, beta_analytical = pretraning_function()

            if mse_error < self.hparams.pretraining_params.initialization_cutoff:
                self.logger.info(f"Pretraining successful on attempt {attempt + 1} with MSE Error: {mse_error}")
                break
            
            self.logger.debug(f"Pretraining attempt {attempt + 1} failed with MSE Error: {mse_error}. Retrying...")
            if mse_error < best_mse:
                best_mse = mse_error
                best_beta = beta_analytical
        else:
            self.logger.info(f"Pretraining did not meet cutoff after {tries} attempts. Using best result with MSE Error: {best_mse}")
            self.y.weight.data = best_beta.T


    def perform_LQR_pretraining(self):
        x = self.sample_inputs(self.hparams.pretraining_params.n_pretraining_colloc)
        Q = self.hparams.problem_params.Q
        R = self.hparams.problem_params.R

        A_np, B_np, S, K = self._linearize_and_solve_care()
        S = torch.tensor(S, device=x.device, dtype=x.dtype)

        self.logger.info(f"{x.shape=}")
        self.logger.info(f"{S.shape=}")
        
        target = (x @ S * x).sum(dim=1) # Target is (1000)
        target = target.unsqueeze(1) # Makes target (1000,1)
        
        self.logger.info(f"{target.shape=}") # Target is (1000)

        # Sanity check to make sure target calculation is correct
        """
        self.logger.info(f"x1 = {x[i, 0]}")
        self.logger.info(f"x2 = {x[i, 1]}")
        self.logger.info(
            f"S11x1^2 + (S12 + S21)x1x2 + S22x2^2 = "
            f"{S[0, 0] * x[i, 0]**2 + (S[0, 1] + S[1, 0]) * (x[i, 0] * x[i, 1]) + S[1, 1] * x[i, 1]**2}"
        )
        self.logger.info(f"Target at index i = {target[i, 0]}")
        
        # Plotting Target Function
        import matplotlib.pyplot as plt

        x_np = x.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy().squeeze()

        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_np[:, 0], x_np[:, 1], target_np, c=target_np, cmap='viridis')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('target')
        ax.set_title('Target vs x1 and x2')
        plt.show()
        """

        return self.set_weights_from_target(x, target)
        

    def perform_xTQX(self):
        x = self.sample_inputs(self.hparams.pretraining_params.n_pretraining_colloc)
        Q = self.hparams.problem_params.Q

        return self.set_weights_from_target(x, self._get_xtqx_target(x))
        

    def set_weights_from_target(self, x: torch.tensor, target: torch.tensor):
        Q = self.hparams.problem_params.Q
        R = self.hparams.problem_params.R
        with torch.no_grad():
            H = self.hidden_layers_output(x)

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
                self.logger.debug(f"x shape: {x.shape}")
                self.logger.debug(f"Q shape: {Q.shape}")
                self.logger.debug(f"R shape: {R.shape}")
                self.logger.debug(f"H shape: {H.shape}")
                self.logger.debug(f"target shape: {target.shape}")
                self.logger.debug(f"HTH shape: {HTH.shape}")
                self.logger.debug(f"HTT shape: {HTT.shape}")
                self.logger.debug(f"I shape: {I.shape}")
                self.logger.debug(f"HTH_reg shape: {HTH_reg.shape}")
                self.logger.debug(f"beta_analytical shape: {beta_analytical.shape}")
                self.logger.debug(f"self.y.weight shape: {self.y.weight.data.shape}")
                self.logger.debug(f"V_approx shape: {V_approx.shape}")

                # self.plot_sample_inputs(x)

            self.logger.info(f"Pretraining completed. MSE Error: {mse_error}")

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