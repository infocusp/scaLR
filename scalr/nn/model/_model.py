"""This file is a base class for the model."""

from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader

import scalr
from scalr.utils import build_object


class ModelBase(nn.Module):
    """Class for the model.
    
    Contains different methods to make a forward() call, load, save weights
    and predict the data provided.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """A function for forward pass of the model to generate outputs."""
        pass

    def load_weights(self, model_weights_path: str):
        """A function to initialize model weights from previous weights."""
        self.load_state_dict(torch.load(model_weights_path)['model_state_dict'])

    def save_weights(self, model_weights_path: str):
        """A function to save model weights at the path."""
        torch.save({'model_state_dict': self.state_dict()}, model_weights_path)

    def get_predictions(self, dl: DataLoader, device: str = 'cpu'):
        """A function to get predictions from the dataloader."""
        pass

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params for model_config."""
        return dict()


def build_model(model_config: dict) -> tuple[nn.Module, dict]:
    """Builder object to get Model, updated model_config."""
    return build_object(scalr.nn.model, model_config)
