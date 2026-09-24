"""This is a test file for genes.py"""

import numpy as np

from scalr.genes import align_genes
from scalr.utils import generate_dummy_anndata


def test_align_genes_full_overlap():
    """When query genes are a superset of reference features, alignment should
    reorder/subset columns exactly to the reference order."""
    adata = generate_dummy_anndata(n_samples=10, n_features=10)
    reference_features = list(adata.var_names[::-1])    # reversed order.

    aligned, report = align_genes(adata, reference_features)

    assert list(aligned.var_names) == reference_features
    assert report.matched_features == len(reference_features)
    assert report.missing_features == []
    assert aligned.shape == (10, len(reference_features))


def test_align_genes_missing_features_are_zero_filled():
    """Reference genes absent from the query should be zero-filled, not dropped."""
    adata = generate_dummy_anndata(n_samples=5, n_features=5)
    reference_features = list(adata.var_names) + ['unseen_gene']

    aligned, report = align_genes(adata, reference_features)

    assert report.missing_features == ['unseen_gene']
    unseen_col = list(aligned.var_names).index('unseen_gene')
    X = aligned.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    assert np.all(X[:, unseen_col] == 0)


def test_align_genes_low_overlap_is_unusable():
    """When gene overlap is below the minimum threshold, the report should say so."""
    adata = generate_dummy_anndata(n_samples=5, n_features=5)
    reference_features = ['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8']

    _, report = align_genes(adata, reference_features, min_feature_overlap=0.9)

    assert not report.is_usable


def test_align_genes_no_overlap_at_all():
    """When none of the reference genes are present, all columns should be zero-filled."""
    adata = generate_dummy_anndata(n_samples=4, n_features=3)
    reference_features = ['unrelated_1', 'unrelated_2']

    aligned, report = align_genes(adata,
                                  reference_features,
                                  min_feature_overlap=0.1)

    assert report.matched_features == 0
    assert report.coverage == 0.0
    assert not report.is_usable
    X = aligned.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    assert np.all(X == 0)


def test_align_genes_preserves_obs():
    """Alignment should carry over the original `obs` metadata unchanged."""
    adata = generate_dummy_anndata(n_samples=6, n_features=4)
    reference_features = list(adata.var_names)

    aligned, _ = align_genes(adata, reference_features)

    assert list(aligned.obs_names) == list(adata.obs_names)
    assert list(aligned.obs.columns) == list(adata.obs.columns)


def test_align_genes_report_str_contains_counts():
    """The report's string form should surface the key coverage numbers."""
    adata = generate_dummy_anndata(n_samples=4, n_features=4)
    reference_features = list(adata.var_names) + ['missing_gene']

    _, report = align_genes(adata, reference_features, min_feature_overlap=0.1)
    text = str(report)

    assert 'Reference features: 5' in text
    assert 'Missing features:      1' in text
