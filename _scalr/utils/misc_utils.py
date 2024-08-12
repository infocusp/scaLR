import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    """To set seed for reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def build_object(module, config: dict):
    name = config.get('name')
    if not name:
        raise ValueError('class name not provided!')

    params = config.get('params', dict())
    default_params = getattr(module, name).get_default_params()
    params = overwrite_default(params, default_params)
    final_config = dict(name=name, params=params)

    return getattr(module, name)(**params), final_config
