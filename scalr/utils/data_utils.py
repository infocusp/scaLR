import numpy as np


def get_one_hot_matrix(labels: np.array):
    """Returns one-hot matrix of given labels.

    Args:
        labels: list of predicted/true labels of each sample.

    Returns:
        one-hot matrix.
    """
    one_hot_matrix = np.zeros((labels.size, labels.max() + 1))
    one_hot_matrix[np.arange(labels.size), labels] = 1

    return one_hot_matrix
