from architectures.pinn import Pinn
from architectures.xtfc_unfreeze import XTFC_Unfreeze
from architectures.xtfc import XTFC
from models.hparams import Hyperparams
from problems.double_integrator import double_integrator
from problems.nonlinear_dynamics import nonlinear_dynamics
from problems.inverted_pendulum import inverted_pendulum
import torch
import log
import logging

if __name__ == "__main__":
    Hyperparams_obj = Hyperparams.from_yaml("yamls/unfreeze_ip.yaml")
    logger = log.get_logger("main")
    logger.setLevel(logging.INFO if Hyperparams_obj.hyper_params.debug == False else logging.DEBUG)
    Hyperparams_obj.logger = logger

    problem = inverted_pendulum(Hyperparams_obj)
    model = XTFC_Unfreeze(problem)
    model.to(device=model.device)

    model.analytical_pretraining()
    model.train_()

    model.plot_trajectory(torch.tensor([[0, 0.1]], dtype=torch.float32, device=model.device), 1, 1000)
    model.plot_value_function()