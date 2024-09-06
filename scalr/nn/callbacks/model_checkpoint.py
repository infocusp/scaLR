"""This file is an implementation of model checkpoint callback."""

import os
from os import path

import torch

from scalr.nn.callbacks import CallbackBase


class ModelCheckpoint(CallbackBase):
    """Model checkpointing to save model weights at regular intervals.

    Attributes:
        epoch: An interger count of epochs trained.
        max_validation_acc: Keeps track of the maximum validation accuracy across all epochs.
        interval: Regular interval of model checkpointing.
    """

    def __init__(self, dirpath: str, interval: int = 5):
        """Intialize required parameters for model checkpoint callback.

        Args:
            dirpath: To store the respective model checkpoints.
            interval: Regular interval of model checkpointing.
        """

        self.epoch = 0
        self.interval = int(interval)
        self.dirpath = dirpath

        if self.interval:
            os.makedirs(path.join(dirpath, 'checkpoints'), exist_ok=True)

    def save_checkpoint(self, model_state_dict: dict, opt_state_dict: dict,
                        path: str):
        """A function to save model & optimizer state dict to the given path.
        
        Args:
            model_state_dict: Model's state dict.
            opt_state_dict: Optimizer's state dict.
            path: Path to store checkpoint to.
        """
        torch.save(
            {
                'epoch': self.epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': opt_state_dict
            }, path)

    def __call__(self, model_state_dict: dict, opt_state_dict: dict, **kwargs):
        """A function that evaluates when to save a checkpoint.

        Args:
            model_state_dict: Model's state dict.
            opt_state_dict: Optimizer's state dict.
        """
        self.epoch += 1
        if self.interval and self.epoch % self.interval == 0:
            self.save_checkpoint(
                model_state_dict, opt_state_dict,
                path.join(self.dirpath, 'checkpoints',
                          f'model_{self.epoch}.pt'))

    @classmethod
    def get_default_params(cls):
        """Class method to get default params for model_config."""
        return dict(dirpath='.', interval=5)
