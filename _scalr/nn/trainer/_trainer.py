import os
from os import path
from time import time

import torch
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from .model.callbacks import CallbackExecutor
from .model import LinearModel


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
        Args:
            model: model to train
            opt_class: optimizer class to train model parameters
            lr: learning rate for optimizer
            weight_decay: L2 penalty for weights
            loss_fn: loss function for training
            callback_params: callback params : dict {'model_checkpoint_interval', 'early_stop_patience', 'early_stop_min_delta'}
            device: device for compuations ('cpu'/'cuda')
            dirpath: dirpath for storing logs, checkpoints, best_model
            model_checkpoint_path: path to resume training from given checkpoint
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
