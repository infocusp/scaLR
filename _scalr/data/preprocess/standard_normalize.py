from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import numpy as np

from _sclar.data.preprocess import Preprocessing

class StandardNorm(Preprocessing):
    
    def __init__(self, **kwargs):
        self.mean = []
        self.std = []
    
    def process_samples(self, data: np.ndarray) -> np.ndarray:
        #preprocess here
        return data
    
    def store_params(self, data: Union[AnnData, AnnCollection], **kwargs) -> None:
        # calculate and update mean and std here
        pass
    
    