from copy import deepcopy

from .file import read_yaml

# TRANSFORMER MODEL CONFIGS ADDITIONS
default_transformer = {
    'transformer_preprocessing': {
        'value_bin': True,
        'n_bins': 51,
        'append_cls': True,
        'include_zero_gene': False,
        'max_len': 3001
    },
    'model': {}
}

# LINEAR MODEL CONFIGS ADDITIONS
default_linear = {'model': {}}

default_feature_selection_config = {
    'chunksize': 2000,
    'method_type': 'feature_chunk',
    'model': {
        'name': 'nn',
        'params': {
            'epochs': 25
        }
    },
    'top_features_stats': {
        'k': 5000,
        'aggregation_strategy': 'mean'
    },
    'store_on_disk': True
}

default_model = {
    'type': None,
    'hyperparameters': {},
    'start_checkpoint': False,
    'resume_from_checkpoint': False
}

default_train_config = {
    'opt': 'adam',
    'loss': 'log',
    'batch_size': 8,
    'lr': 0.001,
    'l2': 0,
    'epochs': 1,
    'callbacks': {
        'model_checkpoint': {
            'interval': 0
        }
    }
}

default_evaluation_config = {
    'batch_size': 8,
    'model_checkpoint': None,
    'metrics': ['accuracy', 'report']
}

# THE DEFAULT CONFIGS TEMPLATE
default_config_ = {
    'device': 'cpu',
    'filepath': '.',
    'exp_name': 'run',
    'exp_run': 0,
    'data': {
        'chunksize': None,
        'normalize_data': False
    }
}


def overwrite_default(user_config: dict, default_config: dict) -> dict:
    """The funnction recursively overwrites information from user_config onto the default_config"""
    for key in user_config.keys():
        if key not in default_config.keys() or not isinstance(
                user_config[key], dict):
            default_config[key] = user_config[key]
        else:
            default_config[key] = overwrite_default(user_config[key],
                                                    default_config[key])

    return default_config


def load_config(path: str) -> dict:
    """This function initializes a default_config file and overwrites information provided by the user.

        Args:
            path: path to config file

        Returns:
            config dict with default parameters loaded and overwritten by user.
    """
    default_config = deepcopy(default_config_)
    user_config = read_yaml(path)

    if 'feature_selection' in user_config:
        default_config['feature_selection'] = default_feature_selection_config

    if 'training' in user_config:
        default_config['training'] = default_train_config
        default_config['model'] = default_model

    if 'evaluation' in user_config:
        default_config['evaluation'] = default_evaluation_config
        default_config['model'] = default_model

    if 'model' in user_config and user_config['model']['type'] == 'transformer':
        default_config['transformer_preprocessing'] = default_transformer[
            'transformer_preprocessing']

    return overwrite_default(user_config, default_config)
