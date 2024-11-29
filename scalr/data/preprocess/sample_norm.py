"""This file performs Sample-wise normalization on the data."""

from typing import Union

import numpy as np

from scalr.data.preprocess import PreprocessorBase


class SampleNorm(PreprocessorBase):
    """Class for Samplewise Normalization"""

    def __init__(self, scaling_factor: float = 1.0):
        """Initialize parameters for Sample-wise normalization.

        Args:
            scaling_factor: `Target sum` to maintain for each sample.
        """

        self.scaling_factor = scaling_factor

    def transform(self, data: np.ndarray) -> np.ndarray:
        """A function to transform provided input data.

        Args:
            data (np.ndarray): Input raw data.

        Returns:
            np.ndarray: Processed data.
        """
        data *= (self.scaling_factor / (data.sum(axis=1).reshape(len(data), 1)))
        return data

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params for preprocess_config."""
        return dict(scaling_factor=1.0)
