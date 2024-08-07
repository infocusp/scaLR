import torch
from torch import nn
from torch import Tensor

from _scalr.nn.model import ModelBase


class SequentialModel(ModelBase):
    """Deep Neural Network model with linear layers."""

    def __init__(self,
                 layers: list[int],
                 dropout: float = 0,
                 activation: str = 'ReLU',
                 weights_init_zero: bool = False):
        """Build Linear Model

        Args:
            layers (list[int]): List of layers' feature size going from
                                                 input_features to output_features.
            dropout (float, optional): dropout after each layer. 
                                       Floating point value [0,1). 
                                       Defaults to 0.
            activation (str, optional): activation function class after each layer.
                                        Defaults to 'ReLU'.
            weights_init_zero (bool, optional): [Bool] to initialize weights of model to zero.
                                                Defaults to False.
        """
        super().__init__()

        try:
            activation = getattr(nn, activation)()
        except:
            raise ValueError(
                f'{activation} is not a valid activation function name in torch.nn'
            )

        dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        n = len(layers)
        for i in range(n - 2):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.layers.append(activation)
            self.layers.append(dropout)
        self.out_layer = nn.Linear(layers[n - 2], layers[n - 1])

        self.weights_init_zero = weights_init_zero
        if weights_init_zero: self.make_weights_zero()

    def make_weights_zero(self):
        for layer in self.layers:
            torch.nn.init.constant_(layer.weight, 0.0)
        torch.nn.init.constant_(self.out_layer.weight, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        """pass input through the network.

            Args:
                x: Tensor, shape [batch_size, layers[0]]

            Returns:
                output dict containing batched layers[-1]-dimensional vectors in ['cls_output'] key.
        """
        output = {}

        for i, layer in enumerate(self.layers):
            x = layer(x)
            output[f'layer{i}_output'] = x

        output['cls_output'] = self.out_layer(x)
        return output

    @classmethod
    def get_default_params(cls):
        """Class method to get default params for model_config"""
        return dict(layers=None,
                    dropout=0,
                    activation='ReLU',
                    weights_init_zero=False)
