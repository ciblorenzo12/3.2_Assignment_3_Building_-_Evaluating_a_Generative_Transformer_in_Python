# I kept random seed setup here so runs stay reproducible.
# It sets the seed for Python, NumPy, and PyTorch.
import random
import numpy as np
import torch

def setGlobalSeed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)