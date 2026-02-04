from architectures.pinn import Pinn
from architectures.xtfc_unfreeze import XTFC_Unfreeze
from architectures.xtfc import XTFC
from models.hparams import Hyperparams
from problems.double_integrator import double_integrator
from problems.nonlinear_dynamics import nonlinear_dynamics
from problems.inverted_pendulum import inverted_pendulum
from models.simulator import Simulator
import torch
import log
import logging
import numpy as np

if __name__ == "__main__":
    Hyperparams_obj = Hyperparams.from_yaml("yamls/unfreeze_ip.yaml")
    logger = log.get_logger("main")
    logger.setLevel(logging.INFO if Hyperparams_obj.hyper_params.debug == False else logging.DEBUG)
    Hyperparams_obj.logger = logger

    np.random.seed(0)
    torch.manual_seed(0)

    problem = inverted_pendulum(Hyperparams_obj)
    model = XTFC_Unfreeze(problem)
    model.to(device=model.device)

    model.analytical_pretraining()
    model.train_model()

    simulator = Simulator(model)
    x0 = torch.tensor([[np.pi - 0.1, 0.0]], dtype=torch.float32, device=model.device)

    simulator.test_model(
        n_points=1,
        t_span=10.0,
        time_step=0.01,
        min_delta=1e-3,
        patience=50,
        random=True,
        ranges = [[-np.pi, np.pi], [-5.0, 5.0]],
        plot=True
    )