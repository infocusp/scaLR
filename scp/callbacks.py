import os

import torch
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    """
    Tensorboard logging of training process.

    Attributes:
        epoch: An interger count of epochs trained.
        writer: object that writes to tensorboard.
    """

    def __init__(self, dirpath: str):
        """
        Args:
            dirpath: path of directory to store the experiment logs
        """
        self.writer = SummaryWriter(dirpath + '/logs')
        self.epoch = 0

    def __call__(self, train_loss: float, train_acc: float, val_loss: float,
                 val_acc: float):
        """
        Logs the train_loss, val_loss, train_accuracy, val_accuracy for each epoch.
        """
        self.epoch += 1
        self.writer.add_scalar('Loss/train', train_loss, self.epoch)
        self.writer.add_scalar('Loss/val', val_loss, self.epoch)
        self.writer.add_scalar('Accuracy/train', train_acc, self.epoch)
        self.writer.add_scalar('Accuracy/val', val_acc, self.epoch)


class EarlyStopping:
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


class ModelCheckpoint:
    """
    Model checkpointing to save model weights at regular intervals and best model.

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
        self.max_validation_acc = float(0)
        self.interval = int(interval)
        self.dirpath = dirpath

        os.makedirs(f'{dirpath}/best_model', exist_ok=True)
        if self.interval:
            os.makedirs(f'{dirpath}/checkpoints', exist_ok=True)

    def save_checkpoint(self, model_state_dict: dict, opt_state_dict: dict,
                        path: str):
        torch.save(
            {
                'epoch': self.epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': opt_state_dict
            }, path)

    def __call__(self, model_state_dict: dict, opt_state_dict: dict,
                 validation_acc: dict):
        self.epoch += 1
        if validation_acc > self.max_validation_acc:
            self.max_validation_acc = validation_acc
            self.save_checkpoint(model_state_dict, opt_state_dict,
                                 f'{self.dirpath}/best_model/model.pt')
        if self.interval and self.epoch % self.interval == 0:
            self.save_checkpoint(
                model_state_dict, opt_state_dict,
                f'{self.dirpath}/checkpoints/model_{self.epoch}.pt')


class CallbackExecutor:
    """
    Wrapper class to incorporate all callbacks implemented
        - TensorboardLogging
        - Early Stopping
        - Checkpointing

    Arguments:
        log: Boolean flag to enable tensorboard logging.
        early_stop: Boolean flag to enable early stopping.
        model_checkpoint: Boolean flag to enable model checkpointing.
    """

    def __init__(self, dirpath: str, callback_paramaters: dict):
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

        self.log = False
        self.early_stop = False
        self.model_checkpoint = False

        if 'tensorboard_logging' in callback_paramaters and callback_paramaters[
                'tensorboard_logging']:
            self.logger = TensorboardLogger(dirpath)
            self.log = True

        if 'model_checkpoint' in callback_paramaters:
            self.checkpoint = ModelCheckpoint(
                dirpath, **callback_paramaters['model_checkpoint'])
            self.model_checkpoint = True

        if 'early_stop' in callback_paramaters:
            self.early_stopper = EarlyStopping(
                **callback_paramaters['early_stop'])
            self.early_stop = True

    def execute(self, model_state_dict: dict, opt_state_dict: dict,
                train_loss: float, train_acc: float, val_loss: float,
                val_acc: float) -> bool:
        """
        Execute all the enabled callbacks. Returns early stopping condition.
        """
        if self.log: self.logger(train_loss, train_acc, val_loss, val_acc)
        if self.model_checkpoint:
            self.checkpoint(model_state_dict, opt_state_dict, val_acc)
        if self.early_stop:
            return self.early_stopper(val_loss)
        else:
            return False
