from abc import ABC, abstractmethod
import os
from os import path

import torch
from torch.utils.tensorboard import SummaryWriter

class Callback(ABC):
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self):
        pass


class CallbackExecutor:
    """
    Wrapper class to incorporate all callbacks implemented
        - TensorboardLogging
        - Checkpointing
        - Early Stopping

    Enabled callbacks are executed with the early stopping callback
    executed last to return a flag for continuation or stopping of model training

    Arguments:
        log: Boolean flag to enable tensorboard logging.
        early_stop: Boolean flag to enable early stopping.
        model_checkpoint: Boolean flag to enable model checkpointing.
    """

    def __init__(self, dirpath: str, callback_params: dict):
        """
        Args:
            dirpath: to store logs and checkpoints
            callback_paramaters: params dict
                - tensorboard_logging: boolean flag
                - model_checkpoint:
                    - interval
                - early_stop:
                    - patience
                    - min_delta
        """

        self.callbacks = []

        for callback in callback_params.keys():
            self.callbacks.append(callback_object)


    def execute(self, model_state_dict: dict, opt_state_dict: dict,
                train_loss: float, train_acc: float, val_loss: float,
                val_acc: float) -> bool:
        """
        Execute all the enabled callbacks. Returns early stopping condition.
        """
        return False
