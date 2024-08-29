from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import numpy as np
from torch import nn

import scalr
from scalr.utils import build_object


class ScoringBase:
    """Base class for scorer"""

    def __init__(self):
        pass

    # Abstract
    def generate_scores(self, model: nn.Module,
                        train_data: Union[AnnData, AnnCollection],
                        val_data: Union[AnnData, AnnCollection], target: str,
                        mappings: dict) -> np.ndarray:
        """Function to return score of each feature for each class

        Args:
            model (nn.Module): trained model to generate scores from
            train_data (Union[AnnData, AnnCollection]): training data of model
            val_data (Union[AnnData, AnnCollection]): validation data of model
            target (str): column in data, used to train the model on
            mappings (dict): mapping of model output dimension to its 
                             corresponding labels in the metadata columns

        Returns:
            np.ndarray: score_matrix [num_classes X num_features]
        """
        pass

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params"""
        return dict()


def build_scorer(scorer_config: dict) -> tuple[ScoringBase, dict]:
    """Builder object to get scorer, updated scorer_config"""
    return build_object(scalr.feature.scoring, scorer_config)
