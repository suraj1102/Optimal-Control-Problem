from hparams import hparams
from train import train, test
from utils import *
import argparse

def main():
    parser = argparse.ArgumentParser(description="Optimal Control Problem Training")
    
    parser.add_argument('--problem', type=str, help='Problem type')
    parser.add_argument('--architecture', type=str, choices=['xtfc', 'xtfc-w-bias', 'xtfc-unfreeze'], help='Model architecture')
    parser.add_argument('--analytical_pretraining', type=str, help='Analytical pretraining method')
    parser.add_argument('--hidden_units', type=int, nargs='+', help='List of hidden units')
    parser.add_argument('--activation', type=str, help='Activation function')
    parser.add_argument('--n_colloc', type=int, help='Number of collocation points')
    parser.add_argument('--input_range', type=float, nargs=2, help='Input range as two floats')
    parser.add_argument('--edge_sampling_weight', type=float, help='Edge sampling weight')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--optimizer', type=str, choices=['ADAM'], help='Optimizer')
    parser.add_argument('--Scheduler', type=str, choices=['None', 'reduce-on-plateau', 'exponential'], help='Scheduler type')
    parser.add_argument('--patience', type=int, help='Patience for reduce-on-plateau scheduler')
    parser.add_argument('--gamma', type=float, help='Gamma for exponential scheduler')
    parser.add_argument('--n_epochs', type=int, help='Number of epochs')
    parser.add_argument('--early_stopping', type=int, help='Early stopping patience (-1 disables)')
    parser.add_argument('--log_wandb', action='store_true', help='Log training to Weights & Biases')
    parser.add_argument('--plot_graphs', action='store_true', help='Plot graphs after training')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model')
    parser.add_argument('--model_save_path', type=str, help='Path to save the trained model')

    args = parser.parse_args()

    flag_hparams = vars(args) # Converts arguments to dictionary
    hparams.update({key: value for key, value in flag_hparams.items() if value is not None})

    if type(hparams['activation']) == str:
        hparams['activation'] = getattr(nn, hparams['activation'])

    model, run, pde_loss, boundary_loss = train(hparams)
    if hparams['plot_graphs']:
        test(model, run)

if __name__ == "__main__":
    main()