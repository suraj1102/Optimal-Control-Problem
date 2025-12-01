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
    'n_epochs': 1_000,
    'early_stopping': 1000, # Indicates patience (in no. of epochs), -1 means no early stopping
    'log_wandb': False,
    'plot_graphs': True,
    'save_model': False,
    'model_save_path': 'trained_model_di.pth',
}
