import yaml
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
default_linear = {
    'model': {}
}

# THE DEFAULT CONFIGS TEMPLATE
default_config = {
    'device': 'cpu',
    'filepath': '.',
    'exp_name': 'run',
    'exp_run': 0,
    'data':{
        'use_top_features': None,
        'store_on_disk': False,
        'load_in_memory': False
    },
    'model': {
        'type': None,
        'hyperparameters': {},
        'start_checkpoint': False,
        'resume_from_checkpoint': False
    },
    'training':{
        'opt': 'adam',
        'loss': 'log',
        'batch_size': 8,
        'lr': 0.001,
        'l2': 0,
        'epochs': 1,
        'callbacks': {
            'early_stop_patience': 3,
            'early_stop_min_delta': 0.0001,
            'model_checkpoint_interval': 5
        }
    },
    'evaluation':{
        'batch_size': 8,
        'model_checkpoint': None,
        'metrics': ['accuracy', 'report']
    }
}

def default(config_, config): 
    """The funnction recursively overwrites information from config_ onto the default config"""
    for key in config_.keys():
        if key not in config.keys() or not isinstance(config_[key], dict):
            config[key] = config_[key]
        else:
            config[key] = default(config_[key], config[key])
            
    return config            

def load_config(path):
    """This function initializes a default config file and overwrites information provided by the user."""
    config = default_config
    config_ = read_yaml(path)
    if 'type' in config_['model'].keys() and config_['model']['type'] == 'transformer':
        config['transformer_preprocessing'] = default_transformer['transformer_preprocessing']
    
    return default(config_, config)
            

        