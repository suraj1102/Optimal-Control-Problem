from dataclasses import dataclass
from typing import List

@dataclass
class HyperHyperParams:
    problem: str
    architecture: str
    analytical_pretraining: str
    log_wandb: bool

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

@dataclass
class Hyperparams:
    hyper_params: HyperHyperParams
    problem_params: ProblemParams
    training_params: TrainingParams
    optimizer_params: OptimizerParams