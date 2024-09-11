"""This file is an implementation of early stopping callback."""

import os
from os import path

import torch

from scalr.nn.callbacks import CallbackBase


class EarlyStopping(CallbackBase):
    """
    Implements early stopping based upon validation loss.

    Attributes:
        patience: Number of epochs with no improvement after which training will be stopped.
        min_delta: Minimum change in the monitored quantity to qualify as an improvement,
        i.e. an absolute change of less than min_delta, will count as no improvement.
    """

    def __init__(self,
                 dirpath: str = None,
                 patience: int = 3,
                 min_delta: float = 1e-4):
        """Intialize required parameters for early stopping callback.

        Args:
            patience: Number of epochs with no improvement after which training will be stopped.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement,
                            i.e. an absolute change of less than min_delta, will count as no improvement.
            epoch: An interger count of epochs trained.
            min_validation_loss: Keeps track of the minimum validation loss across all epochs.
        """
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.epoch = 0
        self.min_val_loss = float('inf')

    def __call__(self, val_loss: float, **kwargs) -> bool:
        """Return `True` if model training needs to be stopped based upon improvement conditions.
        Else returns `False` for continued training.
        """
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.epoch = 0
        elif val_loss >= (self.min_val_loss + self.min_delta):
            self.epoch += 1
            if self.epoch >= self.patience:
                return True
        return False

    @classmethod
    def get_default_params(cls):
        """Class method to get default params for model_config."""
        return dict(patience=3, min_delta=1e-4)
