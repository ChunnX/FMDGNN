import torch
import numpy as np
import os
import random
import pickle
from scipy.io import loadmat

from config import *


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


def split_data(sizes, n_instances=N_SUBJECTS):
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


def get_source_target_domain(path, num_domains, hospital_idx):
    """Return a list contains 6 tensors representing 6 views of one hospital"""
    source_target_domain = []
    for view_num in range(num_domains):
        features = []
        for id in hospital_idx:
            data_path = os.path.join(path, f"data{id}.mat")
            data = loadmat(data_path)["Tensor"][:,:,0,0,0,view_num]
            tensor = torch.Tensor(data)
            feature = vectorize(tensor) # tensor [192, 595]
            features.append(feature)
        # stack the features of each subject into one tensor
        feature_vec = torch.stack(features).squeeze(1)
        # append into a list representing 6 views
        source_target_domain.append(feature_vec)

    return source_target_domain


def save_cross_results(filename, result):
    with open(f"{filename}.pkl", "wb") as file:
        pickle.dump(result, file)


def load_cross_results(filename):
    with open(f"{filename}.pkl", "rb") as file:
        result = pickle.load(file)

    return result

