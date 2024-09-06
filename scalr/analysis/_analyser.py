"""This file is a base class for the analysis module."""

import os
from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
from torch import nn
from torch.utils.data import DataLoader

import scalr
from scalr.utils import build_object


class AnalysisBase:
    """A base class for downstream analysis tasks.
    
    This class provides common attributes and methods for all the analysis tasks.
    It is intended to be subclassed to create task-specific analysis.
    """

    def __init__(self):
        pass

    # Abstract
    def generate_analysis(self, model: nn.Module,
                          test_data: Union[AnnData, AnnCollection],
                          test_dl: DataLoader, dirpath: str, **kwargs):
        """A function to generate analysis, should be overridden by all subclasses.

        Args:
            model (nn.Module): final trained model.
            test_data (Union[AnnData, AnnCollection]): test data to run analysis on.
            test_dl (DataLoader): DataLoader object to prepare inputs for the model.
            dirpath (str): dirpath to store analysis.
            **kwargs: contains all previous analysis done to be used later.
        """
        pass

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params for analysis_config."""
        return dict()


def build_analyser(analysis_config: dict) -> tuple[AnalysisBase, dict]:
    """Builder object to get analyser, updated analyser_config."""
    return build_object(scalr.analysis, analysis_config)
