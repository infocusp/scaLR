from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import torch
from torch import nn
from torch import Tensor

import _scalr
from _scalr.utils import build_object


class ModelBase(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass

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

    def load_weights(self, model_weights_path: str):
        """method to initialize model weights from previous weights"""
        self.load_state_dict(
            torch.load(model_weights_path)['model_state_dict'])

    def save_weights(self, model_weights_path: str):
        """method to save model weights at path"""
        torch.save({'model_state_dict': self.state_dict()}, model_weights_path)

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params for model_config"""
        return dict()


def build_model(model_config):
    return build_object(_scalr.nn.model, model_config)
