from utils import * 

hparams = {
    'problem': 'double-integrator',
    'architecture': 'xtfc', # xtfc, xtfc-w-bias, xtfc-unfreeze, pinn
    'analytical_pretraining': 'xTQx',
    'hidden_units': [128],
    'activation': nn.SiLU,
    'n_colloc': 5_000,
    'input_range': (-1, 1),
    'edge_sampling_weight': 0.2,
    'lr': 1e-3,
    'optimizer': 'ADAM',
    'Scheduler': 'None',
    'patience': 100, # For reduce-on-plateau scheduler
    'gamma': 0.99, # For exponential scheduler
    'n_epochs': 10_000,
    'early_stopping': 500, # Indicates patience (in no. of epochs), -1 means no early stopping
    'log_wandb': False,
    'plot_graphs': True,
    'save_model': True,
    'model_save_path': 'trained_model.pth',
}
