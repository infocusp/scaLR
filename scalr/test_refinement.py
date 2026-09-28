"""This is a test file for refinement.py"""

import numpy as np
import pytest

from scalr.refinement import detect_cluster_key
from scalr.refinement import refine_with_clusters
from scalr.result import PredictionResult
from scalr.utils import generate_dummy_anndata


def _result_for(adata, labels, confidence):
    n = len(adata)
    probabilities = np.zeros((n, 2))
    return PredictionResult(
        obs_names=list(adata.obs_names),
        labels=labels,
        probabilities=probabilities,
        class_names=['A', 'B'],
        confidence=np.array(confidence),
        entropy=np.zeros(n),
        margin=np.zeros(n),
        is_unknown=np.zeros(n, dtype=bool),
    )


def test_detect_cluster_key_finds_known_column():
    """A leiden column should be auto-detected as the cluster key."""
    adata = generate_dummy_anndata(n_samples=4, n_features=3)
    adata.obs['leiden'] = ['0', '0', '1', '1']
    assert detect_cluster_key(adata) == 'leiden'


def test_detect_cluster_key_returns_none_when_absent():
    """No cluster-like column should return None, not raise."""
    adata = generate_dummy_anndata(n_samples=4, n_features=3)
    assert detect_cluster_key(adata) is None


def test_refine_with_clusters_applies_majority_vote():
    """A cluster where one label dominates by confidence should be relabeled uniformly."""
    adata = generate_dummy_anndata(n_samples=4, n_features=3)
    adata.obs['cluster'] = ['0', '0', '0', '1']
    # cluster 0: 2xA (high conf) + 1xB (low conf) -> majority A.
    result = _result_for(adata, ['A', 'A', 'B', 'B'],
                         confidence=[0.9, 0.9, 0.1, 0.5])

    refined = refine_with_clusters(result,
                                   adata,
                                   cluster_key='cluster',
                                   min_cluster_agreement=0.5)

    assert refined.labels[:3] == ['A', 'A', 'A']
    assert refined.labels[3] == 'B'    # cluster 1 has only one cell, unchanged.


def test_refine_with_clusters_preserves_raw_labels():
    """Refinement must keep the pre-refinement labels in metadata['raw_labels']."""
    adata = generate_dummy_anndata(n_samples=3, n_features=3)
    adata.obs['cluster'] = ['0', '0', '0']
    result = _result_for(adata, ['A', 'A', 'B'], confidence=[0.9, 0.9, 0.1])

    refined = refine_with_clusters(result, adata, cluster_key='cluster')

    assert refined.metadata['raw_labels'] == ['A', 'A', 'B']
    # Original result object must not be mutated.
    assert result.labels == ['A', 'A', 'B']


def test_refine_with_clusters_auto_detection():
    """cluster_key='auto' should find a 'louvain' column automatically."""
    adata = generate_dummy_anndata(n_samples=3, n_features=3)
    adata.obs['louvain'] = ['0', '0', '0']
    result = _result_for(adata, ['A', 'A', 'B'], confidence=[0.9, 0.9, 0.1])

    refined = refine_with_clusters(result, adata, cluster_key='auto')
    assert refined.metadata['cluster_key'] == 'louvain'


def test_refine_with_clusters_raises_without_cluster_column():
    """auto detection with no cluster-like column should raise a clear error."""
    adata = generate_dummy_anndata(n_samples=3, n_features=3)
    result = _result_for(adata, ['A', 'A', 'B'], confidence=[0.9, 0.9, 0.1])

    with pytest.raises(ValueError):
        refine_with_clusters(result, adata, cluster_key='auto')


def test_refine_with_clusters_raises_on_mismatched_obs_names():
    """Mismatched obs_names between result and adata should raise, not silently misalign."""
    adata = generate_dummy_anndata(n_samples=3, n_features=3)
    adata.obs['cluster'] = ['0', '0', '0']
    result = _result_for(adata, ['A', 'A', 'B'], confidence=[0.9, 0.9, 0.1])
    result.obs_names = ['x', 'y', 'z']

    with pytest.raises(ValueError):
        refine_with_clusters(result, adata, cluster_key='cluster')
