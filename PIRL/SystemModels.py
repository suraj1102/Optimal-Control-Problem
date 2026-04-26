import numpy as np


class PendulumModel:
    def __init__(self, g=1.0, m=1.0, l=1.0):

        self.g = g
        self.m = m
        self.l = l

    def F(self, x, u):
        """
        x: (n, 3) array-like (cos_th, sin_th, th_dot)
        u: (n, 1) or (n,) array-like (torque)
        Returns: (n, 3) array (d_cos, d_sin, th_accel)
        """
        x = np.asarray(x)
        u = np.asarray(u)
        if u.ndim == 2 and u.shape[1] == 1:
            u = u[:, 0]
        cos_th = x[:, 0]
        sin_th = x[:, 1]
        th_dot = x[:, 2]
        th_accel = 3 * self.g / (2 * self.l) * sin_th + 3.0 / (self.m * self.l**2) * u
        d_cos = -sin_th * th_dot
        d_sin = cos_th * th_dot
        return np.stack([d_cos, d_sin, th_accel], axis=1)

