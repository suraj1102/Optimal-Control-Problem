from train import train, test
from hparams import hparams
from torch import nn

## CONSTS
hparams['log_wandb'] = True
hparams['save_model'] = False
hparams['plot_graphs'] = False
hparams['save_plot'] = False


## HYPERPARAMETER GRID
hidden_units_options = [[10], [50], [100], [400], [10, 10], [30, 30]]
activation_options = [nn.SiLU, nn.ReLU, nn.Tanh, nn.Sigmoid]
analytical_pretraining_options = ['xTQx', 'None', 'LQR']
architecture_options = ['xtfc', 'xtfc-w-bias', 'xtfc-unfreeze', 'pinn']

for hidden_units in hidden_units_options:
    for activation in activation_options:
        for analytical_pretraining in analytical_pretraining_options:
            for architecture in architecture_options:
                # Update hyperparameters
                hparams['hidden_units'] = hidden_units
                hparams['activation'] = activation
                hparams['analytical_pretraining'] = analytical_pretraining
                hparams['architecture'] = architecture

                print(f"Training with hidden_units={hidden_units}, activation={activation.__name__}, "
                      f"analytical_pretraining={analytical_pretraining}, architecture={architecture}")

                # Train and evaluate the model with the current set of hyperparameters
                model = train(hparams)
                test(model, None, hparams)  # Assuming 'run' is not needed for this context