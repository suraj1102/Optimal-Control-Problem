from utils import * 

hparams = {
    'problem': 'double-integrator',
    'architecture': 'xtfc', # xtfc, xtfc-w-bias, xtfc-unfreeze, pinn
    'analytical_pretraining': 'xTQx',
    'hidden_units': [50],
    'activation': nn.SiLU,
    'n_colloc': 5_000,
    'input_range': (-1, 1),
    'edge_sampling_weight': 0.3,
    'lr': 1e-3,
    'optimizer': 'ADAM', # ADAM or LBFGS
    'Scheduler': 'reduce-on-plateau',
    'patience': 100, # For reduce-on-plateau scheduler
    'gamma': 0.99, # For exponential scheduler
    'n_epochs': 5_000,
    'early_stopping': -1, # Indicates patience (in no. of epochs), -1 means no early stopping
    'log_wandb': False,
    'plot_graphs': True,
    'save_model': False,
    'model_save_path': 'trained_model_di.pth',
}
