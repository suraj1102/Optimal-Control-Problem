import torch


class PendulumModel:
    def __init__(self, g=10.0, m=1.0, l=1.0):

        self.g = g
        self.m = m
        self.l = l

    def F(self, x, u):
        """
        x: (n, 3) torch tensor (cos_th, sin_th, th_dot)
        u: (n, 1) or (n,) torch tensor (torque)
        Returns: (n, 3) torch tensor (next_cos_th, next_sin_th, next_th_dot)
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if not torch.is_tensor(u):
            u = torch.tensor(u, dtype=torch.float32)
        if u.ndim == 2 and u.shape[1] == 1:
            u = u[:, 0]
        sin_th = x[:, 0]
        cos_th = x[:, 1]
        th_dot = x[:, 2]
        theta = torch.atan2(sin_th, cos_th)

        # Set defaults for gymnasium params if not present
        dt = torch.tensor(getattr(self, 'dt', 0.05), dtype=torch.float32, device=x.device)
        max_torque = getattr(self, 'max_torque', 2.0)
        max_speed = getattr(self, 'max_speed', 8.0)

        u = torch.clamp(u, -max_torque, max_torque)
        new_th_dot = th_dot + (3 * self.g / (2 * self.l) * torch.sin(theta) + 3.0 / (self.m * self.l**2) * u) * dt
        new_th_dot = torch.clamp(new_th_dot, -max_speed, max_speed)
        new_theta = theta + new_th_dot * dt

        next_sin_th = torch.sin(new_theta)
        next_cos_th = torch.cos(new_theta)
        return torch.stack([next_sin_th, next_cos_th, new_th_dot], dim=1)

