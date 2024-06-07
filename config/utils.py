from copy import deepcopy

from scalr.utils.file import read_yaml
from .default_config import data_config, data_split_config, feature_selection_config, train_config, model_config, evaluation_config

# THE DEFAULT CONFIGS TEMPLATE
default_config_template = {
    'device': 'cuda',
    'filepath': '.',
    'exp_name': 'scalr_test',
    'exp_run': 0
}


def overwrite_default(user_config: dict, default_config: dict) -> dict:
    """The function recursively overwrites information from user_config onto the default_config"""
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
    default_config = deepcopy(default_config_template)
    user_config = read_yaml(path)

    default_config['data'] = data_config
    if 'data_split' in user_config['data'] or 'full_datapath' in user_config['data']:
        default_config['data']['data_split'] = data_split_config

    if 'feature_selection' in user_config:
        default_config['feature_selection'] = feature_selection_config

    if 'training' in user_config:
        default_config['training'] = train_config
        default_config['model'] = model_config

    if 'evaluation' in user_config:
        default_config['evaluation'] = evaluation_config
        default_config['model'] = model_config

    return overwrite_default(user_config, default_config)
