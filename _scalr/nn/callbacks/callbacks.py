import os
from os import path
from typing import Union
from anndata import AnnData
from anndata.experimental import AnnCollection
import _scalr
from _scalr.utils import build_object


class CallbackBase:
    """Base class to build callbacks"""

    def __init__(self, dirpath='.'):
        """Use to generate nessecary arguments or
        create directories
        """
        pass

    def __call__(self):
        """Execute the callback here"""
        pass

    @classmethod
    def get_default_params(cls):
        """Class method to get default params for callbacks config"""
        return None


class CallbackExecutor:
    """
    Wrapper class to execute all enabled callbacks
    
    Enabled callbacks are executed with the early stopping callback
    executed last to return a flag for continuation or stopping of model training
    """

    def __init__(self, dirpath: str, callbacks: list[dict]):
        """
        Args:
            dirpath: to store logs and checkpoints
            callback: list containing multiple callbacks
        """

        self.callbacks = []

        for callback in callbacks:
            if callback.get('params'): callback['params']['dirpath'] = dirpath
            else: callback['params'] = dict(dirpath=dirpath)
            self.callbacks.append(build_object(_scalr.nn.callbacks, callback))

    def execute(self, **kwargs) -> bool:
        """
        Execute all the enabled callbacks. Returns early stopping condition.
        """

        early_stop = False
        for callback in self.callbacks:
            # Below `| False` is to handle cases when callbacks returns None.
            # And we want to return true when early stopping is achieved
            early_stop |= callback(**kwargs) or False

        return early_stop
