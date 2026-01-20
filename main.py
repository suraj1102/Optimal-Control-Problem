from architectures.pinn import Pinn
from architectures.xtfc_unfreeze import XTFC_Unfreeze
from architectures.xtfc import XTFC
from models.hparams import Hyperparams
from problems.double_integrator import double_integrator
from problems.nonlinear_dynamics import nonlinear_dynamics
import torch

if __name__ == "__main__":
    Hyperparams_obj = Hyperparams.from_yaml("yamls/unfreeze.yaml")
    problem = double_integrator(Hyperparams_obj)
    model = XTFC_Unfreeze(problem)
    model.to(device=model.device)

    model.xTQx_analytical_pretraining()
    model.train_()

    model.plot_trajectory(torch.tensor([[1.0, 1.0]], dtype=torch.float32, device=model.device), 1, 1000)
    model.plot_value_function()