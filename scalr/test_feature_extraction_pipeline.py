"""This is a test file for feature_extraction_pipeline.py's `feature_scoring` step."""

import numpy as np

from scalr.feature.scoring import ScoringBase
import scalr.feature.scoring as scoring_module
from scalr.feature_extraction_pipeline import FeatureExtractionPipeline
from scalr.utils import generate_dummy_dge_anndata


class _TaggedFakeScorer(ScoringBase):
    """Returns a score block filled with the model's own tag, so we can check
    each chunk's output ends up at the right position after (possibly
    out-of-order) parallel completion."""

    def generate_scores(self, model, train_data, val_data, target, mappings):
        n_classes = len(mappings[target]['id2label'])
        n_features = train_data.shape[1]
        return np.full((n_classes, n_features), fill_value=model.tag)

    @classmethod
    def get_default_params(cls):
        return dict()


class _Tag:
    """Minimal stand-in for a trained chunked model: just carries an identity tag."""

    def __init__(self, tag):
        self.tag = tag


def _build_pipeline(tmp_path, num_workers, monkeypatch):
    monkeypatch.setattr(scoring_module,
                        '_TaggedFakeScorer',
                        _TaggedFakeScorer,
                        raising=False)

    adata = generate_dummy_dge_anndata(n_donors=4,
                                       cell_type_list=['B_cell', 'T_cell'],
                                       cell_replicate=5,
                                       n_vars=12)
    mappings = {
        'cell_type': {
            'id2label': {
                0: 'B_cell',
                1: 'T_cell'
            },
            'label2id': {
                'B_cell': 0,
                'T_cell': 1
            },
        }
    }

    pipeline = FeatureExtractionPipeline(
        feature_selection_config={
            'feature_subsetsize': 4,
            'scoring_config': {
                'name': '_TaggedFakeScorer'
            },
        },
        dirpath=str(tmp_path),
        device='cpu',
    )
    pipeline.set_data_and_targets(adata, adata, 'cell_type', mappings)
    pipeline.feature_subsetsize = 4
    pipeline.num_workers = num_workers
    pipeline.set_model([_Tag(0), _Tag(1), _Tag(2)])
    return pipeline


def test_feature_scoring_preserves_chunk_order_single_worker(
        tmp_path, monkeypatch):
    """With num_workers=1, each feature chunk's score block should land at its own columns."""
    pipeline = _build_pipeline(tmp_path, num_workers=1, monkeypatch=monkeypatch)

    score_matrix = pipeline.feature_scoring()

    assert score_matrix.shape == (2, 12)
    assert (score_matrix.iloc[:, 0:4] == 0).all().all()
    assert (score_matrix.iloc[:, 4:8] == 1).all().all()
    assert (score_matrix.iloc[:, 8:12] == 2).all().all()


def test_set_data_and_targets_stores_sample_chunksize(tmp_path):
    """set_data_and_targets should retain sample_chunksize, not just accept and drop it."""
    adata = generate_dummy_dge_anndata(n_donors=2,
                                       cell_type_list=['B_cell'],
                                       cell_replicate=2,
                                       n_vars=4)
    mappings = {
        'cell_type': {
            'id2label': {
                0: 'B_cell'
            },
            'label2id': {
                'B_cell': 0
            }
        }
    }

    pipeline = FeatureExtractionPipeline(feature_selection_config={},
                                         dirpath=str(tmp_path),
                                         device='cpu')
    pipeline.set_data_and_targets(adata,
                                  adata,
                                  'cell_type',
                                  mappings,
                                  sample_chunksize=500)

    assert pipeline.sample_chunksize == 500


def test_feature_scoring_preserves_chunk_order_multiple_workers(
        tmp_path, monkeypatch):
    """Parallel (threaded) scoring across chunks should still preserve column order."""
    pipeline = _build_pipeline(tmp_path, num_workers=3, monkeypatch=monkeypatch)

    score_matrix = pipeline.feature_scoring()

    assert score_matrix.shape == (2, 12)
    assert (score_matrix.iloc[:, 0:4] == 0).all().all()
    assert (score_matrix.iloc[:, 4:8] == 1).all().all()
    assert (score_matrix.iloc[:, 8:12] == 2).all().all()
