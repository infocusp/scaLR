import torch
from torch import nn
from torch import Tensor

from _scalr.nn.model import ModelBase


class LinearModel(ModelBase):
    """Deep Neural Network model with linear layers."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_layers: list[int] = None,
                 dropout: float = 0,
                 activation: str = 'ReLU',
                 weights_init_zero: bool = False):
        """Build Linear Model

        Args:
            in_features (int): _description_
            out_features (int): _description_
            hidden_layers (list[int], optional): List of hidden layers' feature size between 
                                                 input_feature and output_features. 
                                                 Defaults to None.
            dropout (float, optional): dropout after each layer. 
                                       Floating point value [0,1). 
                                       Defaults to 0.
            activation (str, optional): activation function class after each layer.
                                        Defaults to 'ReLU'.
            weights_init_zero (bool, optional): [Bool] to initialize weights of model to zero.
                                                Defaults to False.
        """
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass

    def load_model_weights(self, model_weights_path):
        pass
    
    @staticmethod
    def set_defaults(self):
        in_features = 
        out_features = 
        hidden_layers: list[int] = None,
        dropout: float = 0,
        activation: str = 'ReLU',
        weights_init_zero: bool = False)
