"""This is a test file for feature_stability.py"""

from scalr.feature_stability import compute_feature_stability
from scalr.utils import generate_dummy_dge_anndata


def test_compute_feature_stability_returns_bounded_report():
    """Stability report should have a bounded Jaccard similarity and n_runs selection counts."""
    adata = generate_dummy_dge_anndata(
        n_donors=6,
        cell_type_list=['B_cell', 'T_cell', 'DC'],
        cell_replicate=8,
        n_vars=15)

    report = compute_feature_stability(adata,
                                       labels_key='cell_type',
                                       top_k=5,
                                       n_runs=4,
                                       subsample_frac=0.8,
                                       seed=0)

    assert report.n_runs == 4
    assert report.top_k == 5
    assert 0.0 <= report.mean_jaccard <= 1.0
    assert (report.selection_frequency <= report.n_runs).all()
    assert (report.selection_frequency >= 1).all()


def test_compute_feature_stability_top_stable_features_respects_min_runs():
    """top_stable_features should only include genes meeting the min_runs threshold."""
    adata = generate_dummy_dge_anndata(
        n_donors=6,
        cell_type_list=['B_cell', 'T_cell', 'DC'],
        cell_replicate=8,
        n_vars=15)
    report = compute_feature_stability(adata,
                                       labels_key='cell_type',
                                       top_k=5,
                                       n_runs=4,
                                       seed=1)

    all_selected = report.top_stable_features(min_runs=1)
    strict_selected = report.top_stable_features(min_runs=report.n_runs)

    assert set(strict_selected) <= set(all_selected)


def test_feature_stability_report_str_contains_summary():
    """The human-readable report should mention the run count and Jaccard summary."""
    adata = generate_dummy_dge_anndata(
        n_donors=6,
        cell_type_list=['B_cell', 'T_cell', 'DC'],
        cell_replicate=8,
        n_vars=15)
    report = compute_feature_stability(adata,
                                       labels_key='cell_type',
                                       top_k=5,
                                       n_runs=3,
                                       seed=2)
    text = str(report)
    assert 'Jaccard' in text
    assert 'runs' in text.lower()
