import numpy as np


def disturb_none(state: np.ndarray) -> np.ndarray:
    return state


def disturb_gaussian(sigma_theta: float = 0.01, sigma_theta_dot: float = 0.01):
    def _fn(state: np.ndarray) -> np.ndarray:
        noise = np.array(
            [
                np.random.normal(0, sigma_theta),
                np.random.normal(0, sigma_theta_dot),
            ],
            dtype=np.float32,
        )
        return state + noise

    return _fn


def disturb_impulse(prob: float = 0.02, magnitude: float = 1.0):
    def _fn(state: np.ndarray) -> np.ndarray:
        if np.random.rand() < prob:
            kick = np.random.choice([-1, 1]) * magnitude
            return state + np.array([0.0, kick], dtype=np.float32)
        return state

    return _fn


def disturb_combined(sigma: float = 0.005, prob: float = 0.01, magnitude: float = 0.5):
    gaussian = disturb_gaussian(sigma, sigma)
    impulse = disturb_impulse(prob, magnitude)

    def _fn(state: np.ndarray) -> np.ndarray:
        return impulse(gaussian(state))

    return _fn


DISTURB_FNS = {
    "none": disturb_none,
    "gaussian": disturb_gaussian(),
    "impulse": disturb_impulse(),
    "combined": disturb_combined(),
}
