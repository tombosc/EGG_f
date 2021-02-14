from functools import reduce
import operator
import torch
import random
import numpy as np
from egg.core.util import find_lengths

def prod(iterable):  # python3.7
    return reduce(operator.mul, iterable, 1)

def count_params(params):
    S = 0
    for p in params:
        S += prod(p.size())
    return S

def set_seed(seed):
    """
    Seeds the RNG in python.random, torch {cpu/cuda}, numpy.
    :param seed: Random seed to be used

    Public copy of core.utils, I don't want to interfere in the core.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def shuffle_message(message, lengths, rng):
    """ Shuffle the words in message.

    message is a batch_size x length tensor. All the tokens after the first eos
    token (a 0) are ignored, all those strictly before are shuffled.
    """
    bs, L = message.size()
    shuffled = torch.zeros_like(message)
    for i, pack in enumerate(zip(message, lengths)):
        row, length = pack
        length = length.item()
        if not(length == L and row[length-1].item() != 0):
            # this ugly mess is because length == L can mean that the last
            # element of the row is 0... or there is no 0 in the row!
            # TODO see if I can change that without out of bounds accesses
            length = length - 1
        permutation = torch.tensor(rng.permutation(length))
        shuffled[i, :length] = row[:length][permutation]
    return shuffled

def dedup_message(message, lengths):
    """ Deduplicate the words in message.

    message is a batch_size x length tensor. All the tokens after the first eos
    token (a 0) are zeroed, all those strictly before are deduplicated.
    """
    bs, L = message.size()
    deduped = torch.zeros_like(message)
    for i, pack in enumerate(zip(message, lengths)):
        row, length = pack
        length = length.item()
        deduped[i, 0] = row[0]
        k = 1
        for j in range(1, length):
            if row[j] != deduped[i, k-1]:
                deduped[i, k] = row[j]
                k += 1
    return deduped
