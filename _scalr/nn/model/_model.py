import torch
from torch import nn
from torch import Tensor

class ModelBase(nn.Module):
    
    def __init__(**params):
        pass

    def forward(self, x: Tensor) -> Tensor:
        pass

    def load_model_weights(self, model_weights_path):
        pass
    
    def save(self, path):
        pass


def build_model(model_config):
    name = model_config['name']
    params = model_config['params']
    
    model = getattr(_scalr.nn.model, name)(**params)
    
    return model