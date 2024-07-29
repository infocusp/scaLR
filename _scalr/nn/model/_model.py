from abc import ABC, abstractmethod

import torch
from torch import nn
from torch import Tensor

from _scalr.model._model_class_mappings.py import MODEL_CLASS_MAPPINGS

class ModelBase(ABC, nn.Module):

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    # should this be abstract? or predefined by us?
    @abstractmethod
    def load_model_weights(self, model_weights_path):
        pass
    
    def save(self, path):
        pass
    
    # ??
    def predict(self, data):
        pass


def model_builder(model_config):
    name = model_config['name']
    params = model_config['params']
    
    model = MODEL_CLASS_MAPPINGS[name](**params)
    
    return model