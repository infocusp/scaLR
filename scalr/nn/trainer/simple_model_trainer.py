import torch
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from scalr.nn.callbacks import CallbackExecutor
from scalr.nn.trainer import TrainerBase


class SimpleModelTrainer(TrainerBase):
    """Simple model trainer. Works with dataloaders which contain all
    input tensors in line with model input, and the last tensor as
    target to train the model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
