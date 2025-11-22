from utils import *
from train import train, early_stopping
from hparams import hparams as default_hparams
import optuna

def objective(trial):
    hparams = default_hparams.copy()

    hparams["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    hparams["n_colloc"] = trial.suggest_int("n_colloc", 1000, 20000)
    hparams["edge_sampling_weight"] = trial.suggest_float("edge_sampling_weight", 0.0, 1.0)

    activation_name = trial.suggest_categorical("activation",[nn.SiLU, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.LeakyReLU, ])
    hparams["activation"] = activation_name
    
    hparams["early_stopping"] = trial.suggest_int("early_stopping", 100, 1000, step=100)

    hidden_dim = trial.suggest_categorical("hidden_units", [[64], [128], [256], [128, 128]])
    hparams["hidden_units"] = hidden_dim

    model, run, pde_loss, boundary_loss = train(hparams)
    early_stopping.best_loss = float('inf')
    early_stopping.epochs_without_improvement = 0

    return pde_loss


def run_study():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1000, show_progress_bar=True)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    run_study()