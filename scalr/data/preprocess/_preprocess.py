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
        # store all params here
        pass

    # Abstract
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Required function to transform a numpy array

        Args:
            data (np.ndarray): raw data

        Returns:
            np.ndarray: processed data
        """
        pass

    def fit(
        self,
        data: Union[AnnData, AnnCollection],
        sample_chunksize: int,
    ) -> None:
        """Applicable only when you need to see entire train data and
        calculate attributes, as required in StdScaler, etc.
        This method should not return anything only used to store
        attributes which will be used by `transform` method
        IMPORTANT to ensure data is read in chunks only

        Args:
            data (Union[AnnData, AnnCollection]): train_data in backed mode
            sample_chunksize (int): number of samples of data that can at most
                                    be loaded in memory
        """
        pass

    def process_data(self, datapath: dict, sample_chunksize: int, dirpath: str):
        """Process the entire data chunkwise and write the processed data
        to disk

        Args:
            datapath (str): datapath to read data from for transformation
            sample_chunksize (int): number of samples in one chunk 
            dirpath (str): dirpath to write the data to
        """
        if not sample_chunksize:
            # TODO
            raise NotImplementedError(
                'Preprocessing does not work without sample chunksize')

        write_chunkwise_data(datapath,
                             sample_chunksize,
                             dirpath,
                             transform=self.transform)


def build_preprocessor(
        preprocessing_config: dict) -> tuple[PreprocessorBase, dict]:
    """Builder object to get processor, updated preprocessing_config"""
    return build_object(scalr.data.preprocess, preprocessing_config)
