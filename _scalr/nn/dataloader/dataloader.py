from anndata.experimental import AnnLoader

import _scalr
from _scalr.utils import build_object


class DataLoaderBase:

    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size

    def collate_fn(self, batch):
        """Given an input anndata of batch_size,
        the collate function creates inputs and outputs.
        It can also be used to perform batch-wise 
        operations.
        """
        pass

    def __call__(self, adata):
        """Returns a Torch DataLoader object"""
        return AnnLoader(adata,
                         batch_size=self.batch_size,
                         collate_fn=lambda batch: self.collate_fn(batch))


def build_dataloader(dataloader_config, adata):
    return build_object(_scalr.nn.dataloader, dataloader_config)(adata)
