"""This file contains functions related to data utility."""

from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch


def get_one_hot_matrix(data: np.array):
    """This function returns a one-hot matrix of given labels.

    Args:
        data: Categorical data of dim 1D or 2D array.

    Returns:
        one-hot matrix.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    ohe = OneHotEncoder().fit(data)
    one_hot_matrix = ohe.transform(data).toarray()

    return one_hot_matrix


def get_random_samples(
    data: Union[AnnData, AnnCollection],
    n_random_samples: int,
) -> torch.tensor:
    """This function returns random N samples from given data.

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
                                             dtype=torch.float32)

    return random_background_data


def generate_dummy_anndata(n_samples, n_features, target_name='celltype'):
    """This function returns anndata object of shape (n_samples, n_features).
    
    It generates random values for target, batch & env from below mentioned choices.
    If you require more columns, you can add them in the below adata.obs without editing
    already existing columns.

    Args:
        n_samples: Number of samples in anndata.
        n_features: Number of features in anndata.
        target_name: Any preferred target name. Default is `celltype`.

    Returns:
        Anndata object.
    """

    # Setting seed for reproducibility.
    np.random.seed(0)

    # Creating anndata object.
    adata = AnnData(X=np.random.rand(n_samples, n_features))
    adata.obs = pd.DataFrame.from_dict({
        target_name: np.random.choice(['B', 'C', 'DC', 'T'], size=n_samples),
        'batch': np.random.choice(['batch1', 'batch2'], size=n_samples),
        'env': np.random.choice(['env1', 'env2', 'env3'], size=n_samples)
    })
    adata.obs.index = adata.obs.index.astype(str)

    return adata
