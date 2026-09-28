"""This is a test file for feature_subsetting.py"""

from scalr.feature.feature_subsetting import FeatureSubsetting
from scalr.utils import generate_dummy_dge_anndata

_MODEL_CONFIG = {'name': 'SequentialModel', 'params': {'layers': [4, 2]}}
_MODEL_TRAIN_CONFIG = {
    'trainer': 'SimpleModelTrainer',
    'dataloader': {
        'name': 'SimpleDataLoader',
        'params': {
            'batch_size': 8
        }
    },
    'optimizer': {
        'name': 'SGD',
        'params': {
            'lr': 0.01
        }
    },
    'loss': {
        'name': 'CrossEntropyLoss'
    },
    'epochs': 1,
}


def _toy_adata():
    return generate_dummy_dge_anndata(n_donors=4,
                                      cell_type_list=['B_cell', 'T_cell'],
                                      cell_replicate=5,
                                      n_vars=4)


def _resolved_model_params(tmp_path, num_workers):
    adata = _toy_adata()
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

    trainer = FeatureSubsetting(
        feature_subsetsize=4,
        chunk_model_config=_MODEL_CONFIG,
        chunk_model_train_config=_MODEL_TRAIN_CONFIG,
        train_data=adata,
        val_data=adata,
        target='cell_type',
        mappings=mappings,
        dirpath=str(tmp_path),
        device='cpu',
        num_workers=num_workers,
        sample_chunksize=100,
    )
    if num_workers > 1:
        trainer.write_feature_subsetted_data()
    trainer.train_chunked_models()
    model_config, _ = trainer.get_updated_configs()
    return model_config['params']


def test_train_chunked_models_resolves_default_params_single_worker(tmp_path):
    """`get_updated_configs` should reflect resolved (default-filled) params
    sourced from each worker's returned config, not just echo the unresolved
    input config."""
    params = _resolved_model_params(tmp_path, num_workers=1)

    assert params['dropout'] == 0
    assert params['activation'] == 'ReLU'
    assert params['weights_init_zero'] is False


def test_train_chunked_models_resolves_default_params_multiple_workers(
        tmp_path):
    """With num_workers>1 (process-based joblib backend), resolved params from a
    worker must still propagate back to the parent process's config, not just
    echo the unresolved input config."""
    params = _resolved_model_params(tmp_path, num_workers=2)

    assert params['dropout'] == 0
    assert params['activation'] == 'ReLU'
    assert params['weights_init_zero'] is False
