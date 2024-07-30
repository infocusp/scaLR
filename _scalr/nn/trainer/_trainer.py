import os
from os import path
from time import time

import torch
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from _scalr.nn.callbacks import CallbackExecutor
from _scalr.nn.model import LinearModel


class TrainerBase:
    """
    Trainer class to train and validate a model from scratch or resume from checkpoint
    """

    def __init__(self,
                 model,
                 opt,
                 loss_fn,
                 callback):
        """
        get the training objects
        """
        pass
    
    
    def train_one_epoch(self, dl: DataLoader) -> (float, float):
        """Trains one epoch

        Args:
            dl: training dataloader

        Returns:
            Train Loss, Train Accuracy
        """
        return total_loss, accuracy


    def validation(self, dl: DataLoader) -> (float, float):
        """ Validates after training one epoch

        Args:
            dl: validation dataloader

        Returns:
            Validation Loss, Validation Accuracy
        """
        
        return total_loss, accuracy

def build_trainer(name, **kwargs):
    
    return getattr(_scalr.nn.trainer, name)(**kwargs)