from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import numpy as np

from _sclar.data.preprocess import Preprocessing

class SampleNorm(Preprocessing):
    
    def __init__(self, **kwargs):
        pass
    
    def process_samples(self, data: np.ndarray) -> np.ndarray:
        #preprocess here
        return data
