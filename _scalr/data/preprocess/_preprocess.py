from abc import ABC, abstractmethod
from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import numpy as np

class Preprocessing(ABC):
    # Store all kwargs required in parameters using init in the subclass
    
    @abstractmethod
    def process_samples(self, data: np.ndarray) -> np.ndarray:
        pass
    
    # Applicable only when you need to see entire data and
    # store some values, as required in StdScaler, etc.
    # This method cannot return anything only used to store
    # attributes which can be used by `process_samples` method
    # IMPORTANT to ensure data is read chunkwise only
    def store_params(self, data: Union[AnnData, AnnCollection], **kwargs) -> None:
        pass
