from architectures.xtfc import XTFC
import torch
from tqdm import tqdm

class XTFC_Unfreeze(XTFC):
    def __init__(self, problem):
        super().__init__(problem)

    def pre_train_step(self):
        # Unfreeze all layers for training (Do Nothing)
        pass