"""This file is a base class for implementation of Callbacks."""

import os
from os import path
from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection

import scalr
from scalr.utils import build_object


class CallbackBase:
    """Base class to build callbacks."""

    def __init__(self, dirpath='.'):
        """Use to generate necessary arguments or create directories."""
        pass

    def __call__(self):
        """Execute the callback here."""
        pass

    @classmethod
    def get_default_params(cls):
        """Class method to get default params for callbacks config."""
        return None


class CallbackExecutor:
    """
    Wrapper class to execute all enabled callbacks.

    Enabled callbacks are executed with the early stopping callback
    executed last to return a flag for continuation or stopping of model training
    """

    def __init__(self, dirpath: str, callbacks: list[dict]):
        """Intialize required parameters for callbacks.

        Args:
            dirpath: Path to store logs and checkpoints.
            callback: List containing multiple callbacks.
        """

        self.callbacks = []

        for callback in callbacks:
            if callback.get('params'):
                callback['params']['dirpath'] = dirpath
            else:
                callback['params'] = dict(dirpath=dirpath)
            callback_object, _ = build_object(scalr.nn.callbacks, callback)
            self.callbacks.append(callback_object)

    def execute(self, **kwargs) -> bool:
        """Execute all the enabled callbacks. Returns early stopping condition."""

        early_stop = False
        for callback in self.callbacks:
            # Below `| False` is to handle cases when callbacks return None.
            # And we want to return true when early stopping is achieved.
            early_stop |= callback(**kwargs) or False

        return early_stop
