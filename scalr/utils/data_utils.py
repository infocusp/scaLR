""" Data realated utility functions."""

import numpy as np
from sklearn.preprocessing import OneHotEncoder


def get_one_hot_matrix(data: np.array):
    """Returns one-hot matrix of given labels.

    Args:
        data: categorical data of dim 1D or 2D array.

    Returns:
        one-hot matrix.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    ohe = OneHotEncoder().fit(data)
    one_hot_matrix = ohe.transform(data).toarray()

    return one_hot_matrix
