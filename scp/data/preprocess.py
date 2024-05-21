from typing import Union

import numpy as np
import anndata as ad
from anndata import AnnData
from anndata.experimental import AnnCollection


def normalize_data(adata: AnnData, scaling_factor: float = 1.0):
    """Normalize each sample in data

    Args:
        adata: AnnData object to normalize
        scaling_factor: factor by which to scale normalized data

    Returns:
        Normalized AnnData
    """
    adata.X /= (adata.X.sum(axis=1).reshape(len(adata), 1))
    adata.X *= scaling_factor
    return adata
