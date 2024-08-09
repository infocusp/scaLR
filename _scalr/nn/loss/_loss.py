from typing import Union
from anndata import AnnData
from anndata.experimental import AnnCollection

import torch
from torch import nn
import _scalr


class CustomLossBase(nn.Module):
    """Base class to implement custom loss functions"""

    def __init__(self):
        super().__init__()
        self.criterion = None

    def forward(self, out, preds):
        """Returns loss betwen outputs and predictions"""
        return self.criterion(out, preds)


def build_loss_fn(loss_config):
    # Logging
    if not loss_config: loss_config = dict()
    name = loss_config.get('name', 'CrossEntropyLoss')
    loss_config['name'] = name
    params = loss_config.get('params', dict())
    if params: loss_config['params'] = params

    loss_fn = getattr(torch.nn, name)(**params)
    return loss_fn, loss_config
