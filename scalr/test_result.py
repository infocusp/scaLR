"""This is a test file for result.py"""

import numpy as np
import pytest

from scalr.result import PredictionResult
from scalr.utils import generate_dummy_anndata


def _toy_result(obs_names):
    n = len(obs_names)
    probabilities = np.tile([0.7, 0.2, 0.1], (n, 1))
    return PredictionResult(
        obs_names=obs_names,
        labels=['A'] * n,
        probabilities=probabilities,
        class_names=['A', 'B', 'C'],
        confidence=probabilities.max(axis=1),
        entropy=np.zeros(n),
        margin=np.full(n, 0.5),
        is_unknown=np.zeros(n, dtype=bool),
    )


def test_len_matches_labels():
    """__len__ should equal the number of predicted cells."""
    result = _toy_result(['c0', 'c1', 'c2'])
    assert len(result) == 3


def test_to_frame_shape_and_index():
    """to_frame should return one row per cell, indexed by obs_names."""
    result = _toy_result(['c0', 'c1'])
    frame = result.to_frame()

    assert list(frame.index) == ['c0', 'c1']
    assert set(frame.columns) == {
        'scalr_pred', 'scalr_confidence', 'scalr_entropy', 'scalr_margin',
        'scalr_unknown'
    }
    assert len(frame) == 2


def test_write_to_adata_success():
    """write_to_adata should populate obs/obsm/uns when obs_names match."""
    adata = generate_dummy_anndata(n_samples=5, n_features=4)
    result = _toy_result(list(adata.obs_names))

    result.write_to_adata(adata)

    assert list(adata.obs['scalr_pred']) == ['A'] * 5
    assert 'scalr_probabilities' in adata.obsm
    assert adata.obsm['scalr_probabilities'].shape == (5, 3)
    assert adata.uns['scalr'] == result.metadata


def test_write_to_adata_mismatched_obs_names_raises():
    """write_to_adata must refuse to write when obs_names are misaligned."""
    adata = generate_dummy_anndata(n_samples=5, n_features=4)
    result = _toy_result(['wrong_' + n for n in adata.obs_names])

    with pytest.raises(ValueError):
        result.write_to_adata(adata)


def test_write_to_adata_custom_prefix():
    """write_to_adata should honor a custom column/key prefix."""
    adata = generate_dummy_anndata(n_samples=3, n_features=4)
    result = _toy_result(list(adata.obs_names))

    result.write_to_adata(adata, prefix='my_model')

    assert 'my_model_pred' in adata.obs.columns
    assert 'my_model_probabilities' in adata.obsm
    assert 'my_model' in adata.uns
