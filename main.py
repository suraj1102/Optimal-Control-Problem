from architectures.pinn import Pinn
from models.hparams import Hyperparams
from problems.double_integrator import double_integrator
import torch

if __name__ == "__main__":
    Hyperparams_obj = Hyperparams.from_yaml("pinn.yaml")
    problem = double_integrator(Hyperparams_obj)
    model = Pinn(problem)

    model.train()

    model.plot_trajectory(torch.tensor([[1.0, 1.0]], dtype=torch.float32), 1, 1000)