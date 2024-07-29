from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import numpy as np
import torch


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


def get_random_samples(
    data: Union[AnnData, AnnCollection],
    n_random_samples: int,
    device: str,
) -> torch.tensor:
    """Returns random N samples from given data.

    Args:
        data: AnnData or AnnCollection object.
        n_random_samples: number of random samples to extract from the data.

    Returns:
        Chosen random samples tensor.
    """

    random_indices = np.random.randint(0, data.shape[0], n_random_samples)
    random_background_data = data[random_indices].X

    if not isinstance(random_background_data, np.ndarray):
        random_background_data = random_background_data.A

    random_background_data = torch.as_tensor(random_background_data,
                                             dtype=torch.float32).to(device)

    return random_background_data
