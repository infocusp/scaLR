"""This file contains functions related to data utility."""

import anndata
import numpy as np
import pandas as pd


def generate_dummy_anndata(n_samples, n_features, target_name='celltype'):
    """This function returns anndata object of shape (n_samples, n_features).
    It generate random values for target, batch & env from below mentioned choices.
    If you require more columns, you can add in below adata.obs without editing
    already existing columns.

    Args:
        n_samples: Number of samples in anndata
        n_features: Number of features in anndata
        target_name: Any prefered target name. Default is `celltype`.

    Returns:
        Anndata object.
    """

    # Setting seed for reproducibility
    np.random.seed(0)

    # Creating anndata object.
    adata = anndata.AnnData(X=np.random.rand(n_samples, n_features))
    adata.obs = pd.DataFrame.from_dict({
        target_name: np.random.choice(['B', 'C', 'DC', 'T'], size=n_samples),
        'batch': np.random.choice(['batch1', 'batch2'], size=n_samples),
        'env': np.random.choice(['env1', 'env2', 'env3'], size=n_samples)
    })
    adata.obs.index = adata.obs.index.astype(str)

    return adata
