import os
from os import path


class CallbackBase:
    """Base class to build callbacks"""
    
    def __init__(self):
        """Use to generate nessecary arguments or
        create directories
        """
        pass
    
    def __call__(self):
        """Execute the callback here"""
        pass


class CallbackExecutor:
    """
    Wrapper class to execute all enabled callbacks
    
    Enabled callbacks are executed with the early stopping callback
    executed last to return a flag for continuation or stopping of model training
    """

    def __init__(self, dirpath: str, callbacks: dict):
        """
        Args:
            dirpath: to store logs and checkpoints
            callback: callbacks dict
                - name: TensorboardLogging
                - name: EarlyStopping
                  params:
                    patience: 3
                    min_delta: 1.0e-4
                - name: ModelCheckpointing
                  params:
                    interval: 5
        """

        self.callbacks = []

        for callback in callbacks:
            self.callbacks.append(build_callback(callback))

    def build_callback():
        # builder class to create callback object
        pass

    def execute(self, model_state_dict: dict, opt_state_dict: dict,
                train_loss: float, train_acc: float, val_loss: float,
                val_acc: float) -> bool:
        """
        Execute all the enabled callbacks. Returns early stopping condition.
        """
        return False
