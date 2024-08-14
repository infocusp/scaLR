import anndata
import numpy as np
import scanpy as sc

from _scalr.data.preprocess import sample_norm


def test_fit():

    # Setting seed for reproducibility
    np.random.seed(0)

    # Anndata object is required for using pipeline normalization functions.
    adata = anndata.AnnData(X=np.random.rand(100, 25))

    # sample-norm required parameter
    target_sum = 5

    # scalr sample-wise normalization
    scalr_sample_norm = sample_norm.SampleNorm(scaling_factor=target_sum)
    # No need to fit() for sample-norm
    scalr_scaled_data = scalr_sample_norm.transform(adata.X)

    # scanpy sample-wise normalization
    scanpy_scaled_data = sc.pp.normalize_total(adata,
                                               target_sum=target_sum,
                                               inplace=False)['X']

    # asserts to check transformed data is having error less than 1e-15 compared to scanpy's transformed data
    assert sum(
        abs(scanpy_scaled_data.flatten() -
            scalr_scaled_data.flatten()).flatten() < 1e-15
    ) == scalr_scaled_data.flatten().shape[
        0], "The sample norm is incorrectly transforming data, please debug code."
