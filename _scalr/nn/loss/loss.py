from typing import Union
from anndata import AnnData
from anndata.experimental import AnnCollection

import torch
from torch import nn
import _scalr
from _scalr.utils import build_object


class CustomLossBase(nn.Module):
    """Base class to implement custom loss functions"""

    def __init__(self):
        super().__init__()
        self.criterion = None

    def update_from_data(self, data: Union[AnnCollection, AnnData],
                         targets: list[str]):
        """To use data to build any part of class
        This method is optional and cannot return anything. 
        It should only be used to create of modify arguments
        Eg. To use data shape to build input and output features, or
        use data to calculate and build weights. It is important to note 
        data should ONLY be read in chunks at a time.

        Args:
            data (Union[AnnCollection, AnnData]): train_data for processing
            targets (list[string]): target columns present in `obs`
        """
        pass

    def forward(self, out, preds):
        """Returns loss betwen outputs and predictions"""
        return self.criterion(out, preds)


def build_loss_fn(name):
    # Logging
    return getattr([torch.nn, _scalr.nn.loss], name)()
