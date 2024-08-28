from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import torch
from torch.nn.functional import pad

from scalr.nn.dataloader import DataLoaderBase


class SimpleDataLoader(DataLoaderBase):
    """Simple DataLoader which converts all adata values to inputs, and target column in metadata
    to output labels

    Returns:
        PyTorch DataLoader object with (X: Tensor [batch_size, features], y: Tensor [batch_size, ])
    """

    def __init__(self,
                 batch_size: int,
                 target: str,
                 mappings: dict,
                 padding: int = None):
        """
        Args:
            batch_size (int): number of samples to be loaded in each batch
            target (str): corresponding metadata name to be treated as
                          training objective in classification.
                          Must be present as a column_name in adata.obs
            mappings (dict): mapping the target name to respective ids
        """
        super().__init__(batch_size, target, mappings)
        self.padding = padding

    def collate_fn(
        self,
        adata_batch: Union[AnnData, AnnCollection],
    ):
        """Given an input anndata of batch_size,
        the collate function creates inputs and outputs

        Args:
            adata_batch (Union[AnnData, AnnCollection]): anndata view object with batch_size samples

        Returns:
            Tuple(x, y): input to model, output from data
        """

        x = adata_batch.X.float()
        if self.padding and x.shape[1] < self.padding:
            x = pad(x, (0, self.padding - x.shape[1]), 'constant', 0.0)
        y = self.get_targets_ids_from_mappings(adata_batch)[0]

        return x, y

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params for model_config"""
        return dict(batch_size=1, target=None, mappings=dict())
