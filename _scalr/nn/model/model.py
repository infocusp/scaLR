import torch
from torch import nn
from torch import Tensor

import _scalr


class ModelBase(nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: Tensor) -> Tensor:
        pass

    def load_model_weights(self, model_weights_path):
        pass

    def save(self, path):
        pass

    def load_defaults():
        pass

def build_model(model_config):
    name = model_config['name']
    params = model_config.get('params',dict())
    default_params = getattr(_scalr.nn.model, name).get_defautls()
    params = overwrite_defautls(params, default_params)
    
    model = getattr(_scalr.nn.model, name)(**params)

    return model
