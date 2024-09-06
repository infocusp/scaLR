"""This file is an implementation of a sequential model."""

from typing import Tuple

import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader

from scalr.nn.model import ModelBase


class SequentialModel(ModelBase):
    """Class for Deep Neural Network model with linear layers."""

    def __init__(self,
                 layers: list[int],
                 dropout: float = 0,
                 activation: str = 'ReLU',
                 weights_init_zero: bool = False):
        """Initialize required parameters for the linear model.

        Args:
            layers (list[int]): List of layers' feature size going from
                                                 input_features to output_features.
            dropout (float, optional): Dropout after each layer.
                                       Floating point value [0,1).
                                       Defaults to 0.
            activation (str, optional): Activation function class after each layer.
                                        Defaults to 'ReLU'.
            weights_init_zero (bool, optional): [Bool] to initialize weights of the model to zero.
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
        if weights_init_zero:
            self.make_weights_zero()

    def make_weights_zero(self):
        """A function to initialize layer weights to 0."""
        for layer in self.layers:
            torch.nn.init.constant_(layer.weight, 0.0)
        torch.nn.init.constant_(self.out_layer.weight, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        """Pass input through the network.

            Args:
                x: Tensor, shape [batch_size, layers[0]].

            Returns:
                Output dict containing batched layers[-1]-dimensional vectors in ['cls_output'] key.
        """
        output = {}

        for i, layer in enumerate(self.layers):
            x = layer(x)
            output[f'layer{i}_output'] = x

        output['cls_output'] = self.out_layer(x)
        return output

    def get_predictions(
            self,
            dl: DataLoader,
            device: str = 'cpu'
    ) -> Tuple[list[int], list[int], list[list[int]]]:
        """A function to get predictions from a model, from the dataloader.

        Args:
            dl (DataLoader): DataLoader object containing samples.
            device (str, optional): Device to run the model on. Defaults to 'cpu'.

        Returns:
            True labels, Predicted labels, Predicted probabilities of all samples
            in the dataloader.
        """
        self.eval()
        test_labels, pred_labels, pred_probabilities = [], [], []

        for batch in dl:
            with torch.no_grad():
                x, y = batch[0].to(device), batch[1].to(device)
                out = self(x)['cls_output']

            test_labels += y.tolist()
            pred_labels += torch.argmax(out, dim=1).tolist()
            pred_probabilities += out.tolist()

        return test_labels, pred_labels, pred_probabilities

    @classmethod
    def get_default_params(cls):
        """Class method to get default params for model_config."""
        return dict(layers=None,
                    dropout=0,
                    activation='ReLU',
                    weights_init_zero=False)
