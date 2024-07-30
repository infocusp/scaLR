from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import numpy as np

class PreprocessorBase:
    """Base class to build preprocessor"""
    
    #REQUIRED
    def process_samples(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """The method called by the pipeline
        
        Args:
            data: raw numpy array
            **kwargs: additional keyword arguments required in processing
        
        Returns: 
            a processed numpy array
        """
        pass
    
    def store_params(self, data: Union[AnnData, AnnCollection], **kwargs) -> None:
        """Applicable only when you need to see entire data and
        store some values, as required in StdScaler, etc.
        This method cannot return anything only used to store
        attributes which can be used by `process_samples` method
        IMPORTANT to ensure data is read in chunks only
        """
        pass

def build_preprocessor(preprocessing_config):
    name = preprocessing_config['name']
    params = preprocessing_config['params']
    
    return getattr(_scalr.data.preprocess, name)(**params)