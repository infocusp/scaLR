data_config = {'sample_chunksize': 10000, 'normalize_samples': False}

data_split_config = {'split_ratio': [7, 1, 2]}

feature_selection_config = {
    'sample_chunksize': 2000,
    'method_type': 'feature_chunk',
    'model': {
        'name': 'nn',
        'params': {
            'epochs': 25,
            'batch_size': 15000,
            'lr': 1.0e-3,
            'weight_decay': 0.1
        }
    },
    'top_features_stats': {
        'k': 5000,
        'aggregation_strategy': 'mean'
    },
    'store_on_disk': True
}

model_config = {
    'type': 'linear',
    'hyperparameters': {
        'dropout': 0,
        'weights_init_zero': False
    },
    'start_checkpoint': None,
    'resume_from_checkpoint': False,
    'batch_correction': False
}

train_config = {
    'opt': 'adam',
    'loss': 'log',
    'batch_size': 5000,
    'lr': 0.001,
    'weight_decay': 0,
    'epochs': 100,
    'callbacks': {
        'tensorboard_logging': True,
        'early_stop': {
            'patience': 3,
            'min_delta': 1.0e-4
        },
        'model_checkpoint': {
            'interval': 5
        }
    }
}

evaluation_config = {'batch_size': 15000, 'model_checkpoint': None}

shap_config = {
    "top_n": 50,
    "batch_size": 1000,
    "background_tensor": 200,
    "heatmap_n_genes": 20,
    "early_stop": {
        "patience": 5,
        "top_genes": 100,
        "threshold": 90,
    }
}
