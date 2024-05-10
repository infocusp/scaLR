import os
from .EarlyStopping import EarlyStopping
from .ModelCheckpoint import ModelCheckpoint
from .Logs import TensorboardLogger

class CallbackExecuter:
    """
    Wrapper class to incorporate all callbacks implemented
        - TensorboardLogging
        - Early Stopping
        - Checkpointing
    """
    def __init__(self, dirpath, callback_paramaters):
        """
        Args:
            dirpath: to store logs and checkpoints
            callback_paramaters: params dict
                - tensorboard_logging:
                - model_checkpoint:
                    - checkpoint_interval
                - early_stop:
                    - stop_patience 
                    - stop_min_delta
        """

        self.log = False
        self.early_stop = False
        self.model_checkpoint = False
        
        if 'tensorboard_logging' in callback_paramaters and callback_paramaters['tensorboard_logging']:
            self.logger = TensorboardLogger(dirpath)
            self.log = True

        if 'model_checkpoint' in callback_paramaters:
            self.checkpoint = ModelCheckpoint(dirpath, **callback_paramaters['model_checkpoint'])
            self.model_checkpoint = True

        if 'early_stop' in callback_paramaters:
            self.early_stopper = EarlyStopping(**callback_paramaters['early_stop'])
            self.early_stop = True

    def execute(self, model_state_dict, opt_state_dict, train_loss, train_acc, val_loss, val_acc):
        
        if self.log: self.logger(train_loss, train_acc, val_loss, val_acc)
        if self.model_checkpoint: self.checkpoint(model_state_dict, opt_state_dict, val_acc)
        if self.early_stop: 
            return self.early_stopper(val_loss)
        else:
            return False

















