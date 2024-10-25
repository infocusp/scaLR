"""This file performs standard scaler normalization on the data."""

from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import numpy as np

from scalr.data.preprocess import PreprocessorBase


class StandardScaler(PreprocessorBase):
    """Class for Standard Normalization"""

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        """Initialize parameters for standard scaler normalization.

        Args:
            with_mean: Mean for standard scaling.
            with_std: Standard deviation for standard scaling.
        """

        self.with_mean = with_mean
        self.with_std = with_std

        # Parameters for standard scaler.
        self.train_mean = None
        self.train_std = None

    def transform(self, data: np.ndarray) -> np.ndarray:
        """A function to transform provided input data.

        Args:
            data (np.ndarray): raw data

        Returns:
            np.ndarray: processed data
        """
        if not self.with_mean:
            train_mean = np.zeros((1, data.shape[1]))
        else:
            train_mean = self.train_mean
        return (data - train_mean) / self.train_std

    def fit(self, data: Union[AnnData, AnnCollection],
            sample_chunksize: int) -> None:
        """This function calculate parameters for standard scaler object from the train data.
        
        Args:
            data: Data to calculate the required parameters of.
            sample_chunksize: Chunks of data that can be loaded into memory at once.
        
        """

        self.calculate_mean(data, sample_chunksize)
        self.calculate_std(data, sample_chunksize)

    def calculate_mean(self, data: Union[AnnData, AnnCollection],
                       sample_chunksize: int) -> None:
        """Function to calculate mean for each feature in the train data
        
        Args:
            data: Data to calculate the mean of.
            sample_chunksize: Chunks of data that can be loaded into memory at once.
            
        Returns:
            Nothing, stores mean per feature of the train data.
            """

        train_sum = np.zeros(data.shape[1]).reshape(1, -1)

        # Iterate through batches of data to get mean statistics
        for i in range(int(np.ceil(data.shape[0] / sample_chunksize))):
            train_sum += data[i * sample_chunksize:i * sample_chunksize +
                              sample_chunksize].X.sum(axis=0)
        self.train_mean = train_sum / data.shape[0]

    def calculate_std(self, data: Union[AnnData, AnnCollection],
                      sample_chunksize: int) -> None:
        """A function to calculate standard deviation for each feature in the train data.
        
        Args:
            data: Data to calculate the standard deviation of
            sample_chunksize: Chunks of data that can be loaded into memory at once.
            
        Returns:
            Nothing, stores standard deviation per feature of the train data.
            """

        # Getting standard deviation of entire train data per feature.
        if self.with_std:
            self.train_std = np.zeros(data.shape[1]).reshape(1, -1)
            # Iterate through batches of data to get std statistics
            for i in range(int(np.ceil(data.shape[0] / sample_chunksize))):
                self.train_std += np.sum(np.power(
                    data[i * sample_chunksize:i * sample_chunksize +
                         sample_chunksize].X - self.train_mean, 2),
                                         axis=0)
            self.train_std /= data.shape[0]
            self.train_std = np.sqrt(self.train_std)

            # Handling cases where standard deviation of feature is 0, replace it with 1.
            self.train_std[self.train_std == 0] = 1
        else:
            # If `with_std` is False, set train_std to 1.
            self.train_std = np.ones((1, data.shape[1]))

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params for preprocess_config."""
        return dict(with_mean=True, with_std=True)
