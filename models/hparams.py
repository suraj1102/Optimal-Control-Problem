from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
import yaml
import numpy as np

@dataclass
class HyperHyperParams:
    problem: str
    architecture: str
    analytical_pretraining: str
    log_wandb: bool
    bias: bool
    training_visualization: bool = False
    debug: bool = False

@dataclass
class ProblemParams:
    in_dim: int
    input_ranges: List[List[float]]
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
    activation: str
    n_colloc: int
    edge_sampling_weight: List[float]
    n_epochs: int

    l1_lambda: float = 0.0
    l2_lambda: float = 0.0

    def __post_init__(self):
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "gelu": nn.GELU,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
            "softplus": nn.Softplus,
            "silu": nn.SiLU
        }

        key = self.activation.lower()
        if key not in activation_map:
            raise ValueError(
                f"Unknown activation '{self.activation}'. "
                f"Available: {list(activation_map.keys())}"
            )
        
        self.activation = activation_map[key]


@dataclass
class PretrainingParams:
    n_pretraining_colloc: int
    lambda_reg: float

    init_limit: int = 1
    initialization_cutoff: float = -1   # -1 means no loop

@dataclass
class OptimizerParams:
    optimizer: str
    lr: float
    scheduler: str = None
    patience: int = -1
    gamma: float = -1.0
    early_stopping: int = -1

class Device:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

class Hyperparams:
    def __init__(
        self,
        hyper_params: HyperHyperParams,
        problem_params: ProblemParams,
        training_params: TrainingParams,
        optimizer_params: OptimizerParams,
        pretraining_params: PretrainingParams = None,
        device: Device = None
    ):
        
        self.hyper_params = hyper_params
        self.problem_params = problem_params
        self.training_params = training_params
        self.optimizer_params = optimizer_params
        self.pretraining_params = pretraining_params
        self.device = device if device is not None else Device()

        self.problem_params.Q = torch.tensor(self.problem_params.Q, dtype=torch.float32, device=self.device.device)
        self.problem_params.R = torch.tensor(self.problem_params.R, dtype=torch.float32, device=self.device.device)

    @classmethod
    def from_yaml(cls, filepath: str):
        with open(filepath, 'r') as file:
            config = yaml.safe_load(file)

        hyper_params = HyperHyperParams(**config['hyper_params'])
        problem_params = ProblemParams(**config['problem_params'])
        training_params = TrainingParams(**config['training_params'])
        optimizer_params = OptimizerParams(**config['optimizer_params'])
        pretraining_params = PretrainingParams(**config['pretraining_params']) if 'pretraining_params' in config else None

        return cls(
            hyper_params=hyper_params,
            problem_params=problem_params,
            training_params=training_params,
            optimizer_params=optimizer_params,
            pretraining_params=pretraining_params
        )