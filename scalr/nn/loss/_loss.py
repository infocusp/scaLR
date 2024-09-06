"""This file is a base class for loss functions."""

from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import torch
from torch import nn

import scalr


class CustomLossBase(nn.Module):
    """Base class to implement custom loss functions."""

    def __init__(self):
        super().__init__()
        self.criterion = None

    def forward(self, out, preds):
        """Returns loss betwen outputs and predictions."""
        return self.criterion(out, preds)


def build_loss_fn(loss_config):
    """Builder object to get Loss function, updated loss_config."""
    name = loss_config.get('name')
    if not name:
        raise ValueError('Loss function not provided')

    params = loss_config.get('params', dict())

    # TODO: Add provision for custom loss object
    loss_fn = getattr(torch.nn, name)(**params)
    return loss_fn, loss_config
