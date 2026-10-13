"""This file is a base class for loss functions."""

from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import torch
from torch import nn

import scalr
from scalr.utils import build_object


class CustomLossBase(nn.Module):
    """Base class to implement custom loss functions.

    Subclass this, set `self.criterion` in `__init__`, and add the subclass
    next to this file (or import it into `scalr/nn/loss/__init__.py`) so
    `build_loss_fn` can find it by name via the `name`/`params` config
    pattern, the same way as models/preprocessors/splitters/etc.
    """

    def __init__(self):
        super().__init__()
        self.criterion = None

    def forward(self, out, preds):
        """Returns loss betwen outputs and predictions."""
        return self.criterion(out, preds)

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params for loss_config."""
        return dict()


def build_loss_fn(loss_config):
    """Builder object to get Loss function, updated loss_config.

    Resolves `name` against custom loss classes registered under
    `scalr.nn.loss` (subclasses of `CustomLossBase`) first, falling back to
    `torch.nn` for built-in losses (e.g. `CrossEntropyLoss`).
    """
    name = loss_config.get('name')
    if not name:
        raise ValueError('Loss function not provided')

    custom_cls = getattr(scalr.nn.loss, name, None)
    if isinstance(custom_cls, type) and issubclass(custom_cls, CustomLossBase):
        return build_object(scalr.nn.loss, loss_config)

    params = loss_config.get('params', dict())
    loss_fn = getattr(torch.nn, name)(**params)
    return loss_fn, loss_config
