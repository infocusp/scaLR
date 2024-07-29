from abc import ABC, abstractmethod

from anndata.experimental import AnnLoader, AnnCollection
from anndata import AnnData
import torch
from torch import nn
from torch import Tensor

from _scalr.model.datalaoder import DATALOADER_CLASS_MAPPINGS

class DataLoaderBase(ABC):

    @abstractmethod
    def collate_fn(self, batch, **kwargs):
        pass
    
    # This method returns a torch DataLoader object
    def __call__(self, adata, batch_size, **kwargs):
        return AnnLoader(adata,
                         batch_size=batch_size,
                         collate_fn=lambda batch: self.collate_fn(
                         batch, **kwargs))
        

def dataloader_builder(name, **kwargs):
    
    dataloader = DATALOADER_CLASS_MAPPINGS[name](**kwargs)
    
    return dataloader