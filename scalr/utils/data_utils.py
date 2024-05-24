import numpy as np


def get_one_hot_metrix(labels: np.array):
    """Return one hot metrix of given labels.

    Args:
        labels: list of predicted/true labels of each sample.

    Returns:
        one-hot metrix.
    """
    one_hot_metrix = np.zeros((labels.size, labels.max() + 1))
    one_hot_metrix[np.arange(labels.size), labels] = 1

    return one_hot_metrix