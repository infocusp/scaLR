"""This is a test file for Sample-wise normalization."""

import scanpy as sc

from scalr.data.preprocess import sample_norm
from scalr.utils import generate_dummy_anndata


def test_transform():
    '''This function tests the transform function of Sample-wise normalization.

    There is no fit() involved in Sample-wise normalization.
    '''

    # Creating an annadata object.
    adata = generate_dummy_anndata(n_samples=100, n_features=25)

    # Sample-wise norm required parameter.
    target_sum = 5

    # scalr sample-wise normalization.
    scalr_sample_norm = sample_norm.SampleNorm(scaling_factor=target_sum)
    # No need to fit() for sample-norm normalization
    scalr_scaled_data = scalr_sample_norm.transform(adata.X)

    # scanpy sample-wise normalization.
    scanpy_scaled_data = sc.pp.normalize_total(adata,
                                               target_sum=target_sum,
                                               inplace=False)['X']

    # asserts to check transformed data having errors less than 1e-15 compared to scanpy's transformed data.
    assert sum(
        abs(scanpy_scaled_data.flatten() -
            scalr_scaled_data.flatten()).flatten() < 1e-15
    ) == scalr_scaled_data.flatten().shape[
        0], "The sample norm is incorrectly transforming data, please debug code."
