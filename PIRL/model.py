import numpy as np

class PendulumModel:
    def __init__(self, g=1.0, m=1.0, l=1.0):
       
        self.g = g
        self.m = m
        self.l = l

    def f(self, x, u):
        cos_th, sin_th, th_dot = x[0], x[1], x[2]
        th_accel = (3 * self.g / (2 * self.l) * sin_th + 
                    3.0 / (self.m * self.l**2) * u)
        d_cos = -sin_th * th_dot
        d_sin = cos_th * th_dot

        return np.array([d_cos, d_sin, th_accel])