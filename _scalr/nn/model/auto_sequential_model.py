from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import torch
from torch import nn
from torch import Tensor

from _scalr.nn.model import SequentialModel


class AutoSequentialModel(SequentialModel):
    """Deep Neural Network model with linear layers."""

    def __init__(self,
                 hidden_layers: list[int] = list(),
                 dropout: float = 0,
                 activation: str = 'ReLU',
                 weights_init_zero: bool = False):
        """Build Linear Model

        Args:
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
        layers = [1] + hidden_layers + [1]
        super().__init__(layers, dropout, activation, weights_init_zero)

    def from_data(self, data: Union[AnnCollection, AnnData],
                  targets: list[str]):
        """To set the input and output features based upon data

        Args:
            data (Union[AnnCollection, AnnData]): train_data for processing
            targets (list[string]): list of targets
        """
        data_in_features = len(data.var_names)
        data_out_features = data.obs.targets[0].nunique()

        if self.layers:
            self.layers[0] = nn.Linear(data_in_features,
                                       self.layers[0].out_features)
            self.out_layer = nn.Linear(self.out_layer.in_features,
                                       data_out_features)
        else:
            self.out_layer = nn.Linear(data_in_features, data_out_features)

        if self.weights_init_zero:
            self.make_weights_zero()

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
        return dict(hidden_layers=None,
                    dropout=0,
                    activation='ReLU',
                    weights_init_zero=False)
