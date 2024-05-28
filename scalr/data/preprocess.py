from typing import Union

import numpy as np


def normalize_data(data: np.ndarray, scaling_factor: float = 1.0):
    """Normalize each sample in data

    Args:
        data: numpy array object to normalize
        scaling_factor: factor by which to scale normalized data

    Returns:
        Normalized numpy array
    """
    data *= (scaling_factor / (data.sum(axis=1).reshape(len(data), 1)))
    return data
