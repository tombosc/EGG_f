from functools import reduce
import operator

def prod(iterable):  # python3.7
    return reduce(operator.mul, iterable, 1)

def count_params(params):
    S = 0
    for p in params:
        print(p.size())
        S += prod(p.size())
    return S
