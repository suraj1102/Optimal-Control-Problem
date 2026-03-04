from models.problem import problem
import numpy as np
import torch
import torch.nn as nn
from models.hparams import Hyperparams

class cart_pole(problem):
    def __init__(self, hparams: Hyperparams):
        super().__init__(hparams)

        self.hparams.problem_params.labels = ['theta', 'thetadot', 'x', 'xdot']
        self.logger = self.hparams.logger

        self.mass_bob = hparams.problem_params.mass_bob
        self.mass_cart = hparams.problem_params.mass_cart
        self.length_rod = hparams.problem_params.length_rod
        self.gravity = hparams.problem_params.gravity
        self.eq_point = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.hparams.device.device)



    def f_x(self, x: torch.Tensor) -> torch.Tensor:
        theta = x[:, 0]
        omega = x[:, 1]
        vel   = x[:, 3]

        m = self.mass_bob
        mc = self.mass_cart
        l = self.length_rod
        g = self.gravity

        M = mc + m
        J = m * l**2

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        den = M * (m*l**2 + J) - (m*l*cos_t)**2

        theta_ddot = (
            M*m*g*l*sin_t + (m*l*cos_t)*(m*l*omega**2*sin_t)
        ) / den

        x_ddot = (
            (-theta_ddot*cos_t + omega*sin_t) * m*l / M
        )

        return torch.stack([
            omega,
            theta_ddot,
            vel,
            x_ddot
        ], dim=1)

    def g_x(self, x: torch.Tensor) -> torch.Tensor:
        theta = x[:, 0]

        m = self.mass_bob
        mc = self.mass_cart
        l = self.length_rod

        M = mc + m
        J = m * l**2

        cos_t = torch.cos(theta)

        den = M * (m*l**2 + J) - (m*l*cos_t)**2

        return torch.stack([
            torch.zeros_like(theta),
            (-m*l*cos_t) / den,
            torch.zeros_like(theta),
            torch.ones_like(theta) / M
        ], dim=1)

    def pde_residual(self, x: torch.Tensor, grad_v: torch.Tensor) -> torch.Tensor:
        # HJB equation residual for cart pole control problem
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        V_x1 = grad_v[:, 0]
        V_x2 = grad_v[:, 1]
        V_x3 = grad_v[:, 2]
        V_x4 = grad_v[:, 3]

        Q = self.hparams.problem_params.Q
        R = self.hparams.problem_params.R

        term1 = V_x1 * x2
        term2 = V_x3 * x4
        term3 = Q[0, 0] * torch.square(x1)
        term4 = Q[1, 1] * torch.square(x2)
        term5 = Q[2, 2] * torch.square(x3)
        term6 = Q[3, 3] * torch.square(x4)
        term7 = -1 * torch.square(V_x4) * -1 * R[0, 0] ** -1 * 2 * self.mass_bob + 2 * self.mass_cart + -1 * self.mass_bob * torch.square(torch.cos(x1)) ** -2 + R[0, 0] ** -1 * self.mass_bob + self.mass_cart ** -1 * 2 * self.mass_bob + 2 * self.mass_cart + -1 * self.mass_bob * torch.square(torch.cos(x1)) ** -1 + self.mass_bob * R[0, 0] ** -1 * self.mass_bob + self.mass_cart ** -1 * 2 * self.mass_bob + 2 * self.mass_cart + -1 * self.mass_bob * torch.square(torch.cos(x1)) ** -2 * torch.square(torch.cos(x1))
        term8 = -1 * V_x2 * V_x4 * -1/2 * self.length_rod ** -1 * R[0, 0] ** -1 * self.mass_bob + self.mass_cart ** -1 * 2 * self.mass_bob + 2 * self.mass_cart + -1 * self.mass_bob * torch.square(torch.cos(x1)) ** -1 * torch.cos(x1) + -1/2 * self.mass_bob * self.length_rod ** -1 * R[0, 0] ** -1 * self.mass_bob + self.mass_cart ** -1 * 2 * self.mass_bob + 2 * self.mass_cart + -1 * self.mass_bob * torch.square(torch.cos(x1)) ** -2 * torch.cos(x1) ** 3
        term9 = -1 * V_x2 * self.length_rod ** -1 * 2 * self.mass_bob + 2 * self.mass_cart + -1 * self.mass_bob * torch.square(torch.cos(x1)) ** -1 * -1 * self.gravity * self.mass_bob * torch.sin(x1) + -1 * self.gravity * self.mass_cart * torch.sin(x1) + -1 * self.length_rod * self.mass_bob * torch.square(x2) * torch.cos(x1) * torch.sin(x1)
        term10 = V_x4 * self.length_rod * self.mass_bob * self.mass_bob + self.mass_cart ** -1 * x2 * torch.sin(x1) + self.length_rod ** -2 * self.mass_bob ** -1 * 2 * self.mass_bob + 2 * self.mass_cart + -1 * self.mass_bob * torch.square(torch.cos(x1)) ** -1 * self.length_rod**2 * self.mass_bob**2 * torch.square(x2) * torch.cos(x1) * torch.sin(x1) + -1 * self.gravity * self.length_rod * self.mass_bob * self.mass_bob + self.mass_cart * torch.sin(x1) * torch.cos(x1)
        term11 = -1/4 * torch.square(V_x2) * self.length_rod ** -2 * R[0, 0] ** -1 * 2 * self.mass_bob + 2 * self.mass_cart + -1 * self.mass_bob * torch.square(torch.cos(x1)) ** -2 * torch.square(torch.cos(x1))

        """
        Term 7 and 11 have very high error contribution
        term7 prob: torch.square(V_x4)
        """

        # if self.hparams.hyper_params.debug:
        #     self.logger.debug(f"term1: {term1}")
        #     self.logger.debug(f"term2: {term2}")
        #     self.logger.debug(f"term3: {term3}")
        #     self.logger.debug(f"term7: {term7}")
        #     self.logger.debug(f"{torch.square(V_x4)=}")
        #     self.logger.debug(f"term8: {term8}")
        #     self.logger.debug(f"term9: {term9}")
        #     self.logger.debug(f"term10: {term10}")
        #     self.logger.debug(f"term11: {term11}")
        #     self.logger.debug(f"{torch.square(V_x2)=}")


        return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11

