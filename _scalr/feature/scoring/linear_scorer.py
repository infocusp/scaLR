from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import numpy as np
import torch
from torch import nn

from _scalr.feature.scoring import ScoringBase


class LinearScorer(ScoringBase):

    def __init__(self):
        pass

    def generate_scores(self, model: nn.Module, *args, **kwargs) -> np.ndarray:
        """Return the weights of model as score
        """
        return model.state_dict()['out_layer.weight'].cpu().detach().numpy()
