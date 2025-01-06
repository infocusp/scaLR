"""This file is a base class for preprocessing module."""

from os import path
from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import numpy as np

import scalr
from scalr.utils import build_object
from scalr.utils import write_chunkwise_data


class PreprocessorBase:
    """Base class for preprocessor"""

    def __init__(self, **kwargs):
        # Store all params here.
        pass

    # Abstract
    def transform(self, data: np.ndarray) -> np.ndarray:
        """A required function to transform a numpy array.

        Args:
            data (np.ndarray): Input raw data.

        Returns:
            np.ndarray: Processed data.
        """
        pass

    def fit(
        self,
        data: Union[AnnData, AnnCollection],
        sample_chunksize: int,
    ) -> None:
        """A function to calculate attributes for transformation.
        
        It is required only when you need to see the entire train data and
        calculate attributes, as required in StdScaler, etc. This method
        should not return anything, it should be used to store attributes
        that will be used by the `transform` method.

        Args:
            data (Union[AnnData, AnnCollection]): train_data in backed mode.
            sample_chunksize (int): several samples of data that can at most
                                    be loaded in memory.
        """
        pass

    def process_data(self,
                     full_data: Union[AnnData, AnnCollection],
                     sample_chunksize: int,
                     dirpath: str,
                     num_workers: int = 1):
        """A function to process the entire data chunkwise and write the processed data
        to disk.

        Args:
            full_data (Union[AnnData, AnnCollection]): Full data for transformation.
            sample_chunksize (int): Number of samples in one chunk.
            dirpath (str): Path to write the data to.
            num_workers (int): number of jobs to run in parallel for data writing.
        """
        if not sample_chunksize:
            # TODO
            raise NotImplementedError(
                'Preprocessing does not work without sample chunk size')

        write_chunkwise_data(full_data,
                             sample_chunksize,
                             dirpath,
                             transform=self.transform,
                             num_workers=num_workers)


def build_preprocessor(
        preprocessing_config: dict) -> tuple[PreprocessorBase, dict]:
    """Builder object to get a processor, updated preprocessing_config."""
    return build_object(scalr.data.preprocess, preprocessing_config)
