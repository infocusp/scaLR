from typing import Union

import numpy as np

from _scalr.data.preprocess import PreprocessorBase


class SampleNorm(PreprocessorBase):

    def __init__(self, scaling_factor: float = 1.0):
        """
        Args:
            scaling_factor: `Target sum` to keep for each sample.
            """

        self.scaling_factor = scaling_factor

    def transform(self, data: np.ndarray) -> np.ndarray:
        """The method called by the pipeline to process a chunk of
        samples.

        Args:
            data (np.ndarray): raw data

        Returns:
            np.ndarray: processed data
        """
        data *= (self.scaling_factor / (data.sum(axis=1).reshape(len(data), 1)))
        return data

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params for preprocess_config"""
        return dict(scaling_factor=1.0)
