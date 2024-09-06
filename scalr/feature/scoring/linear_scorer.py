"""This file is a implementation of linear scorer."""

from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import numpy as np
import torch
from torch import nn

from scalr.feature.scoring import ScoringBase


class LinearScorer(ScoringBase):
    """Class for linear scorer.
    
    This Scorer is only applicable for linear (single-layer) models.
    It directly uses the weights as score for each feature.
    """

    def __init__(self):
        pass

    def generate_scores(self, model: nn.Module, *args, **kwargs) -> np.ndarray:
        """A function to generate and return the weights of model as a score."""
        return model.state_dict()['out_layer.weight'].cpu().detach().numpy()
