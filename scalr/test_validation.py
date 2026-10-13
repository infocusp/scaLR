"""This is a test file for validation.py"""

from anndata import AnnData
import numpy as np

from scalr.utils import generate_dummy_anndata
from scalr.validation import detect_normalization_state
from scalr.validation import validate


def test_validate_usable_data():
    """A valid AnnData with a labels_key should validate as usable with no errors."""
    adata = generate_dummy_anndata(n_samples=50, n_features=20)
    report = validate(adata, labels_key='celltype')
    assert report.is_usable
    assert report.errors == []


def test_validate_missing_labels_key():
    """A missing labels_key should be reported as an error."""
    adata = generate_dummy_anndata(n_samples=20, n_features=10)
    report = validate(adata, labels_key='does_not_exist')
    assert not report.is_usable
    assert any('does_not_exist' in e for e in report.errors)


def test_validate_duplicate_obs_names():
    """Duplicate cell IDs should be reported as an error."""
    adata = generate_dummy_anndata(n_samples=10, n_features=5)
    adata.obs_names = ['cell_0'] * len(adata)
    report = validate(adata)
    assert not report.is_usable
    assert any(
        'duplicate cell IDs' in e.lower() or 'duplicate cell' in e.lower()
        for e in report.errors)


def test_validate_nan_values():
    """NaN values in X should be reported as an error."""
    adata = generate_dummy_anndata(n_samples=10, n_features=5)
    adata.X[0, 0] = np.nan
    report = validate(adata)
    assert not report.is_usable
    assert any('nan' in e.lower() for e in report.errors)


def test_validate_gene_overlap_report():
    """Gene coverage against model_features should be reported."""
    adata = generate_dummy_anndata(n_samples=10, n_features=10)
    model_features = list(
        adata.var_names[:5]) + ['missing_gene_1', 'missing_gene_2']
    report = validate(adata,
                      model_features=model_features,
                      min_feature_overlap=0.1)
    assert report.is_usable
    assert any('missing' in w.lower() for w in report.warnings)


def test_validate_low_gene_overlap_is_error():
    """Gene coverage below `min_feature_overlap` should make data unusable."""
    adata = generate_dummy_anndata(n_samples=10, n_features=10)
    model_features = ['missing_gene_1', 'missing_gene_2', 'missing_gene_3']
    report = validate(adata,
                      model_features=model_features,
                      min_feature_overlap=0.5)
    assert not report.is_usable


def test_detect_normalization_state_raw_counts():
    """Integer-valued, large-magnitude data should be detected as raw counts."""
    adata = generate_dummy_anndata(n_samples=20, n_features=10)
    adata.X = np.random.randint(0, 500, size=adata.X.shape).astype(float)
    assert detect_normalization_state(adata) == 'raw_counts'


def test_detect_normalization_state_scaled():
    """Data containing negative values should be detected as scaled/standardized."""
    adata = generate_dummy_anndata(n_samples=20, n_features=10)
    adata.X = adata.X - 5
    assert detect_normalization_state(adata) == 'scaled'


def test_validate_rejects_non_anndata_input():
    """A non-AnnData input should produce a single, clear error."""
    report = validate({'not': 'an anndata'})
    assert not report.is_usable
    assert len(report.errors) == 1


def test_validate_empty_adata_is_error():
    """An AnnData with 0 observations should be reported as an error."""
    adata = AnnData(X=np.zeros((0, 5)))
    report = validate(adata)
    assert not report.is_usable
    assert any('0 observations' in e for e in report.errors)


def test_validate_missing_group_key_is_error():
    """A missing group_key should be reported as an error."""
    adata = generate_dummy_anndata(n_samples=10, n_features=5)
    report = validate(adata, group_key='does_not_exist')
    assert not report.is_usable
    assert any('does_not_exist' in e for e in report.errors)


def test_validate_duplicate_var_names_is_warning_not_error():
    """Duplicate gene IDs should warn but not block usage."""
    adata = generate_dummy_anndata(n_samples=10, n_features=4)
    adata.var_names = ['g0', 'g0', 'g1', 'g2']
    report = validate(adata)
    assert report.is_usable
    assert any('duplicate gene' in w.lower() for w in report.warnings)


def test_validation_report_str_reflects_status():
    """The human-readable report should mention 'unusable' when there are errors."""
    adata = generate_dummy_anndata(n_samples=10, n_features=5)
    report = validate(adata, labels_key='does_not_exist')
    assert 'Status: unusable' in str(report)
