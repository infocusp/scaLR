import os
from .EarlyStopping import EarlyStopping
from .ModelCheckpoint import ModelCheckpoint
from .Logs import Logs

class CallBack:
    """
    Wrapper class to incorporate all callbacks implemented
        - Logging
        - Early Stopping
        - Checkpointing
    """
    def __init__(self, filepath, callbacks, model):
        """
        Args:
            filepath: to store logs and checkpoints
            callbacks: params dict {'model_checkpoint_interval', 'early_stop_patience', 'early_stop_min_delta'}
        """
        checkpoint_interval = callbacks['model_checkpoint_interval'] 
        stop_patience = callbacks['early_stop_patience'] 
        stop_min_delta = callbacks['early_stop_min_delta']
        
        os.makedirs(f'{filepath}/checkpoints', exist_ok=True)
        self.logger = Logs(filepath)
        self.early_stopper = EarlyStopping(stop_patience, stop_min_delta)
        self.checkpoint = ModelCheckpoint(filepath, checkpoint_interval, model)

    def __call__(self, model_state_dict, opt_state_dict, train_loss, train_acc, val_loss, val_acc):
        self.logger(train_loss, train_acc, val_loss, val_acc)
        self.checkpoint(model_state_dict, opt_state_dict, val_acc)
        return self.early_stopper(val_loss)

















