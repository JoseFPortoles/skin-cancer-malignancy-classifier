import random
import numpy as np
import torch
import os

def seed_all(seed: int):
    """Seed every relevant library with the same seed.

    Args:
        seed (int): Seed.
    """    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)