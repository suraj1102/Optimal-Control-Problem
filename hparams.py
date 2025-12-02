import numpy as np
from utils import * 

hparams = {
    'problem': 'inverted-pendulum',
    'architecture': 'xtfc-unfreeze', # xtfc, xtfc-w-bias, xtfc-unfreeze, pinn
    'analytical_pretraining': 'xTQx', # None, xTQx
    'hidden_units': [50],
    'activation': nn.SiLU,
    'n_colloc': 5_000,
    'input_range': (-np.pi/6, np.pi/6),
    'edge_sampling_weight': 0.3,
    'lr': 1e-3,
    'optimizer': 'ADAM', # ADAM or LBFGS
    'Scheduler': 'None',
    'patience': 100, # For reduce-on-plateau scheduler
    'gamma': 0.99, # For exponential scheduler
    'n_epochs': 20_000,
    'early_stopping': -1, # Indicates patience (in no. of epochs), -1 means no early stopping
    'log_wandb': False,
    'plot_graphs': True,
    'save_model': False,
    'model_save_path': 'trained_model_di.pth',
    'save_plot': True,

    'Q': np.matrix([[100.0, 0.0], [0.0, 1.0]]),
    'R': np.matrix([[1.0]]),
    'mass': 0.2,
    'length': 1.0,
    'gravity': 9.81,
    'LOAD_MODEL': False,
}
