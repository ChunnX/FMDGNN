import torch
import numpy as np
import os
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def vectorize(matrix):
    """Extract the feature vector vertically from the adjacency matrix"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = matrix.shape[0]
    assert matrix.shape == (m, m)
    vec = torch.Tensor(matrix[np.tril_indices(m, k=-1)])
    return vec.clone().detach().requires_grad_(False).reshape(1, -1).to(device)


def antiVectorize(vec):
    """Restore the adjacency matrix from the feature vector"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = np.zeros((35, 35))
    M[np.tril_indices(35, k=-1)] = vec.detach().cpu()
    M = torch.Tensor(M + M.T)
    return M.clone().detach().requires_grad_(False).to(device)


def split_data(sizes, n_instances=678):
    """generate a random permutation of indices from 0 to n_instances"""
    shuffled_indices = np.random.permutation(n_instances)
    groups = []
    start = 0
    for size in sizes:
        groups.append(shuffled_indices[start:start+size])
        start += size
    return groups


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



