import os
from os import path

import torch
from torch.utils.tensorboard import SummaryWriter

from _scalr.model.callbacks import Callback


class TensorboardLogger(Callback):
    """
    Tensorboard logging of training process.

    Attributes:
        epoch: An interger count of epochs trained.
        writer: object that writes to tensorboard.
    """

    def __init__(self, dirpath: str = '.'):
        """
        Args:
            dirpath: path of directory to store the experiment logs
        """
        self.writer = SummaryWriter(path.join(dirpath, 'logs'))
        self.epoch = 0

    def __call__(self, train_loss: float, train_acc: float, val_loss: float,
                 val_acc: float):
        """
        Logs the train_loss, val_loss, train_accuracy, val_accuracy for each epoch.
        """
        self.epoch += 1
        self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, self.epoch)
        self.writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, self.epoch)
