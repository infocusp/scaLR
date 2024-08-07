import os
from os import path

import torch

from _scalr.nn.callbacks import CallbackBase


class EarlyStopping(CallbackBase):
    """
    Implements early stopping based upon validation loss

    Attributes:
        patience: number of epochs with no improvement after which training will be stopped
        min_delta: Minimum change in the monitored quantity to qualify as an improvement,
                            i.e. an absolute change of less than min_delta, will count as no improvement.
    """

    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        """
        Args:
            patience: number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in the monitored quantity to qualify as an improvement,
                            i.e. an absolute change of less than min_delta, will count as no improvement.
            epoch: An interger count of epochs trained.
            min_validation_loss: keeps the track of the minimum validation loss across all epochs.
        """
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.epoch = 0
        self.min_validation_loss = float('inf')

    def __call__(self, validation_loss: float) -> bool:
        """
        Return `True` if model training needs to be stopped based upon improvement conditions. Else returns
        `False` for continued training.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.epoch = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.epoch += 1
            if self.epoch >= self.patience:
                return True
        return False
    
    @classmethod
    def get_default_params(cls):
        """Class method to get default params for model_config"""
        return dict(patience=3,
                    min_delta=1e-4)