from typing import Union

import anndata as ad
from anndata import AnnData
from anndata.experimental import AnnCollection
import numpy as np


def normalize_data(adata: AnnData, scaling_factor: float = 1.0):
    """Normalize each sample in data

    Args:
        adata: AnnData object to normalize
        scaling_factor: factor by which to scale normalized data

    Returns:
        Normalized AnnData
    """
    adata.X *= (scaling_factor / (adata.X.sum(axis=1).reshape(len(adata), 1)))
    return adata
