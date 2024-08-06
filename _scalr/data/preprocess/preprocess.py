from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import numpy as np

import _scalr
from _scalr.utils import write_chunkwise_data


class PreprocessorBase:
    """Base class to build preprocessor"""

    def __init__(self, **kwargs):
        # store all params here
        pass

    #REQUIRED
    def process_samples(self, data: np.ndarray) -> np.ndarray:
        """The method called by the pipeline to process a chunk of
        samples.

        Args:
            data (np.ndarray): raw data

        Returns:
            np.ndarray: processed data
        """
        pass

    def generate_transform_attributes(
        self,
        data: Union[AnnData, AnnCollection],
    ) -> None:
        """Applicable only when you need to see entire train data and
        calculate attributes, as required in StdScaler, etc.
        This method should not return anything only used to store
        attributes which will be used by `process_samples` method
        IMPORTANT to ensure data is read in chunks only

        Args:
            data (Union[AnnData, AnnCollection]): train_data in backed mode
        """
        pass

    def process_data(self, datapaths: str, sample_chunksize: int,
                     dirpaths: str):
        """Process each split of the data chunkwise

        Args:
            datapaths (str): _description_
            sample_chunksize (int): _description_
            dirpaths (str): _description_
        """
        if not sample_chunksize:
            raise NotImplementedError(
                'Preprocessing does not work without sample chunksize')

        for split in datapaths.keys():
            write_chunkwise_data(datapaths[split],
                                 sample_chunksize,
                                 dirpaths[split],
                                 transform=self.process_samples)


def build_preprocessor(preprocessing_config):
    name = preprocessing_config['name']
    params = preprocessing_config.get('params', list())

    return getattr(_scalr.data.preprocess, name)(**params)
