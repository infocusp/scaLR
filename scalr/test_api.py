"""This is an end-to-end test file for the simple public API (api.py, artifact.py)."""

import numpy as np
import pytest

import scalr
from scalr.utils import generate_dummy_dge_anndata


def _toy_adata():
    return generate_dummy_dge_anndata(n_donors=8,
                                      cell_type_list=['B_cell', 'T_cell', 'DC'],
                                      cell_replicate=10,
                                      n_vars=20)


def test_train_returns_annotation_model_with_metrics():
    """scalr.train should return a calibrated model with held-out metrics."""
    adata = _toy_adata()

    model = scalr.train(adata,
                        labels_key='cell_type',
                        group_key='donor_id',
                        hidden_layers=(16,),
                        epochs=2,
                        batch_size=32,
                        device='cpu',
                        verbose=False)

    assert set(model.class_names) == {'B_cell', 'T_cell', 'DC'}
    assert model.features == list(adata.var_names)
    assert 'macro_f1' in model.metrics
    assert 0.0 <= model.metrics['macro_f1'] <= 1.0


def test_annotate_result_is_aligned_and_typed():
    """scalr.annotate should return a PredictionResult aligned to obs_names."""
    adata = _toy_adata()
    model = scalr.train(adata,
                        labels_key='cell_type',
                        group_key='donor_id',
                        hidden_layers=(16,),
                        epochs=2,
                        batch_size=32,
                        device='cpu',
                        verbose=False)

    result = scalr.annotate(adata, model=model, device='cpu')

    assert isinstance(result, scalr.PredictionResult)
    assert result.obs_names == list(adata.obs_names)
    assert len(result.labels) == len(adata)
    assert result.probabilities.shape == (len(adata), 3)
    assert all(l in model.class_names for l in result.labels)
    assert result.is_unknown.dtype == bool


def test_model_save_and_load_roundtrip(tmp_path):
    """A saved model artifact should reload and produce identical predictions."""
    adata = _toy_adata()
    model = scalr.train(adata,
                        labels_key='cell_type',
                        group_key='donor_id',
                        hidden_layers=(16,),
                        epochs=2,
                        batch_size=32,
                        device='cpu',
                        verbose=False)

    artifact_dir = tmp_path / 'pbmc_v1'
    model.save(str(artifact_dir))

    assert (artifact_dir / 'manifest.json').exists()
    assert (artifact_dir / 'model.pt').exists()
    assert (artifact_dir / 'features.json').exists()
    assert (artifact_dir / 'label_mapping.json').exists()

    reloaded = scalr.load_model(str(artifact_dir))
    assert reloaded.class_names == model.class_names
    assert reloaded.features == model.features

    original_result = model.predict(adata, device='cpu')
    reloaded_result = reloaded.predict(adata, device='cpu')

    assert np.allclose(original_result.probabilities,
                       reloaded_result.probabilities,
                       atol=1e-5)
    assert original_result.labels == reloaded_result.labels


def test_annotate_from_artifact_path(tmp_path):
    """scalr.annotate should accept a model artifact directory path directly."""
    adata = _toy_adata()
    model = scalr.train(adata,
                        labels_key='cell_type',
                        group_key='donor_id',
                        hidden_layers=(16,),
                        epochs=2,
                        batch_size=32,
                        device='cpu',
                        verbose=False)
    artifact_dir = tmp_path / 'pbmc_v1'
    model.save(str(artifact_dir))

    result = scalr.annotate(adata, model=str(artifact_dir), device='cpu')
    assert len(result.labels) == len(adata)


def test_annotate_with_gene_subset_query():
    """annotate should align a query with fewer/reordered genes automatically."""
    adata = _toy_adata()
    model = scalr.train(adata,
                        labels_key='cell_type',
                        group_key='donor_id',
                        hidden_layers=(16,),
                        epochs=2,
                        batch_size=32,
                        device='cpu',
                        verbose=False)

    # Query with a shuffled subset of genes (still above min_feature_overlap).
    subset_genes = list(adata.var_names[:15])[::-1]
    query = adata[:, subset_genes].copy()

    result = scalr.annotate(query, model=model, device='cpu',
                            min_feature_overlap=0.1)
    assert len(result.labels) == len(query)
    assert result.metadata['gene_coverage'] < 1.0


def test_predict_raises_when_gene_overlap_too_low():
    """predict should refuse to score a query with too little gene overlap."""
    adata = _toy_adata()
    model = scalr.train(adata,
                        labels_key='cell_type',
                        group_key='donor_id',
                        hidden_layers=(16,),
                        epochs=2,
                        batch_size=32,
                        device='cpu',
                        verbose=False)

    unrelated_query = adata[:, adata.var_names[:1]].copy()
    unrelated_query.var_names = ['completely_unrelated_gene']

    with pytest.raises(ValueError):
        model.predict(unrelated_query, device='cpu', min_feature_overlap=0.5)


def test_annotate_rejects_unsupported_preprocess_mode():
    """annotate should raise NotImplementedError for unsupported preprocess modes."""
    adata = _toy_adata()
    model = scalr.train(adata,
                        labels_key='cell_type',
                        group_key='donor_id',
                        hidden_layers=(16,),
                        epochs=2,
                        batch_size=32,
                        device='cpu',
                        verbose=False)

    with pytest.raises(NotImplementedError):
        scalr.annotate(adata, model=model, device='cpu', preprocess='force')


def test_predict_open_set_false_never_flags_unknown():
    """With open_set=False, no cell should be flagged unknown regardless of confidence."""
    adata = _toy_adata()
    model = scalr.train(adata,
                        labels_key='cell_type',
                        group_key='donor_id',
                        hidden_layers=(16,),
                        epochs=2,
                        batch_size=32,
                        device='cpu',
                        verbose=False)

    result = model.predict(adata, device='cpu', open_set=False)
    assert not result.is_unknown.any()


def test_train_without_group_key_still_produces_usable_model():
    """Training without a group_key should fall back to stratified splitting and still work."""
    adata = _toy_adata()
    model = scalr.train(adata,
                        labels_key='cell_type',
                        group_key=None,
                        hidden_layers=(16,),
                        epochs=2,
                        batch_size=32,
                        device='cpu',
                        verbose=False)

    result = model.predict(adata, device='cpu')
    assert len(result.labels) == len(adata)


def test_load_model_rejects_newer_format_version(tmp_path):
    """Loading an artifact with a newer format_version than supported should raise."""
    adata = _toy_adata()
    model = scalr.train(adata,
                        labels_key='cell_type',
                        group_key='donor_id',
                        hidden_layers=(16,),
                        epochs=2,
                        batch_size=32,
                        device='cpu',
                        verbose=False)
    artifact_dir = tmp_path / 'pbmc_v1'
    model.save(str(artifact_dir))

    from scalr.utils import read_data
    from scalr.utils import write_data
    manifest_path = artifact_dir / 'manifest.json'
    manifest = read_data(str(manifest_path))
    manifest['format_version'] = 999
    write_data(manifest, str(manifest_path))

    with pytest.raises(ValueError):
        scalr.load_model(str(artifact_dir))
