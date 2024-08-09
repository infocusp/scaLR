import os
from os import path

import torch

from _scalr.nn.callbacks import CallbackBase


class ModelCheckpoint(CallbackBase):
    """
    Model checkpointing to save model weights at regular intervals

    Attributes:
        epoch: An interger count of epochs trained.
        max_validation_acc: keeps the track of the maximum validation accuracy across all epochs.
        interval: regular interval of model checkpointing.
    """

    def __init__(self, dirpath: str, interval: int = 5):
        """
        Args:
            dirpath: to store the respective model checkpoints
            interval: regular interval of model checkpointing
        """

        self.epoch = 0
        self.interval = int(interval)
        self.dirpath = dirpath

        if self.interval:
            os.makedirs(path.join(dirpath, 'checkpoints'), exist_ok=True)

    def save_checkpoint(self, model_state_dict: dict, opt_state_dict: dict,
                        path: str):
        torch.save(
            {
                'epoch': self.epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': opt_state_dict
            }, path)

    def __call__(self, model_state_dict: dict, opt_state_dict: dict, **kwargs):
        self.epoch += 1
        if self.interval and self.epoch % self.interval == 0:
            self.save_checkpoint(
                model_state_dict, opt_state_dict,
                path.join(self.dirpath, 'checkpoints',
                          f'model_{self.epoch}.pt'))

    @classmethod
    def get_default_params(cls):
        """Class method to get default params for model_config"""
        return dict(dirpath='.', interval=5)
