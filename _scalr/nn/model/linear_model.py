import torch
from torch import nn
from torch import Tensor

from _scalr.nn.model import ModelBase

class LinearModel(ModelBase):
    """Deep Neural Network model with linear layers."""

    def __init__(self):
        """
        Args:
            layers: List of layers' feature size going from input_ft -> out_ft. eg. [22858, 2048, 6] (req)
            dropout: dropout after each layer. Floating point value [0,1)
            activation_class: activation function class after each layer
            weights_init_zero: [Bool] to initialize weights of model to zero
        """
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass
    
    def load_model_weights(self, model_weights_path):
        pass
