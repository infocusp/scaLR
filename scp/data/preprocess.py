from typing import Union

import numpy as np
import anndata as ad
from anndata import AnnData
from anndata.experimental import AnnCollection

def _normalize_subdata(adata: AnnData, scaling_factor: float = 1.0):
    """Normalize each sample in subset of data

    Args:
        adata: AnnData object to normalize
        scaling_factor: factor by which to scale normalized data

    Returns:
        Normalized AnnData
    """
    adata.X = (adata.X / (adata.X.sum(axis=1).reshape(len(adata), 1))) * scaling_factor
    return adata

def normalize_data(adata: Union[AnnData, AnnCollection], chunksize:int = None, scaling_factor: float = 1.0):
    
    # for start in range(0, len(adata), chunksize):
    if chunksize is None:
        return _normalize_subdata(adata=adata, scaling_factor=scaling_factor)
        
    