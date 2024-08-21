from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import numpy as np
from torch import nn

import _scalr
from _scalr.utils import build_object


class ScoringBase:

    def __init__(self):
        pass

    def generate_scores(self, model: nn.Module,
                        train_data: Union[AnnData, AnnCollection],
                        val_data: Union[AnnData, AnnCollection], target: str,
                        mappings: dict) -> np.ndarray:
        """Executor function, to return score of each feature for each class
        score: (num_classes X num_features)
        """
        return

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params"""
        return dict()


def build_scorer(scorer_config):
    return build_object(_scalr.feature.scoring, scorer_config)
