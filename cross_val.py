import numpy as np


def k_fold_split(hospital_idx, n_split):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(hospital_idx))
    split_indices = np.array_split(shuffled_indices, n_split)

    return split_indices