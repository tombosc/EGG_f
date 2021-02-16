from functools import reduce
import operator
import torch
import numpy as np

def prod(iterable):  # python3.7
    return reduce(operator.mul, iterable, 1)

def count_params(params):
    S = 0
    for p in params:
        S += prod(p.size())
    return S

def set_torch_seed(seed):
    """
    Seeds the RNG in torch {cpu/cuda}
    :param seed: Random seed to be used

    Public copy of core.utils, I don't want to interfere in the core.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

