"""This is a test file for drift.py (dataset-level drift detection)"""

from anndata import AnnData
import numpy as np
import pytest
from scipy import sparse

from scalr.drift import compute_reference_stats
from scalr.drift import detect_drift


def _adata(X, n_genes=4):
    features = [f'gene_{i}' for i in range(n_genes)]
    return AnnData(X=sparse.csr_matrix(X)), features


def test_compute_reference_stats_matches_manual_mean_std():
    """Reference gene mean/std should match plain numpy over the same matrix."""
    X = np.array([[1.0, 0.0, 3.0, 0.0], [1.0, 2.0, 3.0, 0.0],
                  [1.0, 4.0, 3.0, 0.0]])
    adata, _ = _adata(X)

    stats = compute_reference_stats(adata)

    np.testing.assert_allclose(stats['gene_mean'], X.mean(axis=0))
    np.testing.assert_allclose(stats['gene_std'], X.std(axis=0))
    assert stats['n_cells'] == 3
    assert stats['library_size_mean'] == pytest.approx(X.sum(axis=1).mean())
    assert stats['n_genes_detected_mean'] == pytest.approx(
        (X > 0).sum(axis=1).mean())


def test_detect_drift_reports_no_drift_for_identical_distribution():
    """Querying with data drawn from the same distribution as the reference should not flag drift."""
    rng = np.random.default_rng(0)
    X_ref = rng.normal(loc=2.0, scale=0.5, size=(200, 5)).clip(min=0)
    X_query = rng.normal(loc=2.0, scale=0.5, size=(200, 5)).clip(min=0)
    features = [f'gene_{i}' for i in range(5)]

    reference = compute_reference_stats(AnnData(X=sparse.csr_matrix(X_ref)))
    query = AnnData(X=sparse.csr_matrix(X_query))

    report = detect_drift(query, reference, features)

    assert report.is_drifted is False
    assert report.drifted_gene_fraction < 0.1


def test_detect_drift_flags_a_systematic_mean_shift():
    """A large, uniform shift in every gene's mean expression should be flagged as drifted."""
    rng = np.random.default_rng(0)
    X_ref = rng.normal(loc=2.0, scale=0.3, size=(200, 5)).clip(min=0)
    X_query = X_ref[:150] + 5.0    # shift every gene far outside its reference std
    features = [f'gene_{i}' for i in range(5)]

    reference = compute_reference_stats(AnnData(X=sparse.csr_matrix(X_ref)))
    query = AnnData(X=sparse.csr_matrix(X_query))

    report = detect_drift(query, reference, features)

    assert report.is_drifted is True
    assert report.drifted_gene_fraction == 1.0
    assert len(report.top_drifted_genes) == 5
    assert report.n_query_cells == 150
    assert report.n_reference_cells == 200
    assert 'DRIFTED' in str(report)


def test_detect_drift_flags_severe_library_size_shift_even_if_genes_look_stable(
):
    """A big library-size/detected-gene shift should flag drift even if per-gene z-scores stay mild."""
    rng = np.random.default_rng(0)
    X_ref = rng.normal(loc=2.0, scale=0.5, size=(200, 5)).clip(min=0)
    # Scale every cell up proportionally: per-gene relative pattern is similar,
    # but total library size/detected genes shift hard.
    X_query = X_ref[:100] * 20.0
    features = [f'gene_{i}' for i in range(5)]

    reference = compute_reference_stats(AnnData(X=sparse.csr_matrix(X_ref)))
    query = AnnData(X=sparse.csr_matrix(X_query))

    report = detect_drift(query, reference, features, severity_z_threshold=3.0)

    assert report.is_drifted is True
    assert abs(report.library_size_z) >= 3.0


def test_detect_drift_rejects_feature_count_mismatch():
    """A query with a different number of columns than the reference should raise, not silently misalign."""
    X_ref = np.ones((10, 4))
    reference = compute_reference_stats(AnnData(X=sparse.csr_matrix(X_ref)))
    query = AnnData(X=sparse.csr_matrix(np.ones((5, 3))))

    with pytest.raises(ValueError):
        detect_drift(query, reference, features=['a', 'b', 'c'])
