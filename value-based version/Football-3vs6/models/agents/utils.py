import numpy as np
import igraph as ig
import inspect
import functools
import torch
import math

def is_acyclic(adjacency):
    prod = np.eye(adjacency.shape[0])
    for _ in range(1, adjacency.shape[0] + 1):
        prod = np.matmul(adjacency, prod)
        if np.trace(prod) != 0:
            return False
    return True


EPSILON = 1e-8
def pruning(G, A):
    with torch.no_grad():
        # Find the smallest threshold that removes all cycle-inducing edges
        thresholds = np.unique(A)
        for step, t in enumerate(thresholds):
            # print("Edges/thresh", G.get_adjacency()._data, t)
            to_keep = A > t + EPSILON
            new_adj = torch.tensor(G.get_adjacency()._data) * to_keep

            if is_acyclic(new_adj):
                # G.get_adjacency()._data.copy_(new_adj)
                new_G = ig.Graph.Weighted_Adjacency(new_adj.tolist())
                break
    return new_G, new_adj

def pruning_1(G, A):
    A = torch.tensor(A)
    with torch.no_grad():
        while not is_acyclic(A):
            A_nonzero = torch.nonzero(A)
            rand_int_index = np.random.randint(0, len(A_nonzero))
            A[A_nonzero[rand_int_index][0]][A_nonzero[rand_int_index][1]] = 0
        new_G = ig.Graph.Weighted_Adjacency(A.tolist())
        return new_G, A


def matrix_poly(matrix, d):
    x = torch.eye(d).double()+ torch.div(matrix, d)
    return torch.matrix_power(x, d)

def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A