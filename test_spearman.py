from scipy.stats import spearmanr, kendalltau
import numpy as np

A = np.asarray([
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 0, 0, 0, 1, 1, 1, 1, 1],
])

s = spearmanr(A[0], A[1])
print(s)
tau = kendalltau(A[0], A[1])
print(tau)


A = np.asarray([
    [1, 2, 3, 4,     5, 6, 7, 8, 9],
    [0, 0, 0.1, 0.1, 1, 1, 1.3, 1.3, 1.3],
])

s = spearmanr(A[0], A[1])
print(s)

tau = kendalltau(A[0], A[1])
print(tau)


