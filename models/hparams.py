from dataclasses import dataclass
from typing import List
import torch

@dataclass
class HyperHyperParams:
    problem: str
    architecture: str
    analytical_pretraining: str
    log_wandb: bool
    bias: bool

@dataclass
class ProblemParams:
    in_dim: int
    input_ranges: List
    mass_bob: float = None
    mass_cart: float = None
    height_cart: float = None
    length_rod: float = None
    gravity: float = None

    Q: List[List[float]] = None
    R: List[List[float]] = None
    

@dataclass
class TrainingParams:
    hidden_units: int
    activation: function
    n_colloc: int
    edge_sampling_weight: List[float]
    n_epochs: int

@dataclass
class OptimizerParams:
    optimizer: str
    lr: float
    scheduler: str
    patience: int
    gamma: float
    early_stopping: int

class Device:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

@dataclass
class Hyperparams:
    hyper_params: HyperHyperParams
    problem_params: ProblemParams
    training_params: TrainingParams
    optimizer_params: OptimizerParams
    device: Device = Device()