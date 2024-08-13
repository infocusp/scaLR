from typing import Union

from absl import logging
import numpy as np

from _scalr.data.preprocess import PreprocessorBase
from _scalr.utils.misc_utils import set_logging_level


class SampleNorm(PreprocessorBase):

    def __init__(self, scaling_factor: float = 1.0):
        """
        Args:
            scaling_factor: `Target sum` to keep for each sample.
            """

        self.scaling_factor = scaling_factor

        set_logging_level('INFO')
        logging.info('Applying Sample-wise normalization on data.')

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
