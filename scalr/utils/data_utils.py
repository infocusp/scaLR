"""This file contains functions related to data utility."""

from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
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


def generate_dummy_dge_anndata(n_donors: int = 5,
                               cell_type_list: list[str] = [
                                   'B_cell', 'T_cell', 'DC'
                               ],
                               cell_replicate: int = 2,
                               n_vars: int = 10) -> AnnData:
    """This function returns anndata object for DGE analysis
    with shape (n_donors*len(cell_type_list)*cell_replicate, n_vars).

    It generates obs with random donors with a fixed clinical condition (disease_x or normal).
    Includes all the cell types in `cell_type_list` with number of `cell_replicate` for each donor.
    It generates a csr(Compressed Sparse Row) matrix with random gene expression values.
    It generates var with random gene name as `var.index` of length `n_vars`.

    Args:
        n_donors: Number of donors or subjects in `anndata.obs`.
        cell_type_list: List of different cell types to include.
        cell_replicate: Number of cell replicates per cell type.
        n_vars: Number of genes to include in `anndata.var`.

    Returns:
        Anndata object.
    """

    # Setting seed for reproducibility.
    np.random.seed(0)

    donor_list = [f'D_{i}' for i in range(1, n_donors + 1)]
    condition_array = np.random.choice(['disease_x', 'normal'],
                                       size=n_donors,
                                       replace=True)

    # Creating obs
    obs_data = []
    for donor, condition in zip(donor_list, condition_array):
        obs_data.extend([{
            'donor_id': donor,
            'cell_type': cell_type,
            'disease': condition
        } for i in range(cell_replicate) for cell_type in cell_type_list])
    obs = pd.DataFrame(obs_data)
    n_obs = obs.shape[0]
    obs.index = obs.index.astype(str)

    # Random geneexpression matrix
    X = csr_matrix(np.random.rand(n_obs, n_vars))

    # Creating var
    var = pd.DataFrame({
        'gene_id': [f'gid_{i}' for i in range(1, n_vars + 1)],
        'gene_name': [f'gene_{i}' for i in range(1, n_vars + 1)]
    }).set_index('gene_name')
    var.index = var.index.astype(str)

    # Creating AnnData object
    adata = AnnData(X=X, obs=obs, var=var)

    return adata
