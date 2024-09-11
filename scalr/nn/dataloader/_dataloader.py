"""This file is a base class for dataloader."""

from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
from anndata.experimental import AnnLoader
import torch
from torch import Tensor

import scalr
from scalr.utils import build_object


class DataLoaderBase:

    def __init__(
        self,
        batch_size: int = 1,
        target: Union[str, list[str]] = None,
        mappings: dict = None,
    ):
        """Initilize required parameters for dataloader.

        Args:
            batch_size (int, optional): _description_. Defaults to 1.
            target ([str, list[str]]): List of target. Defaults to None.
            mappings (dict): List of label mappings of each target to. Defaults to None.
        """
        self.batch_size = batch_size
        self.target = target
        self.mappings = mappings

    # Abstract
    def collate_fn(self, batch):
        """Collate function for dataloader. Should be implemented in child classes.
        
        Given an input anndata of batch_size, the collate function creates inputs and outputs.
        It can also be used to perform batch-wise operations.
        """
        pass

    def get_targets_ids_from_mappings(
            self, adata: Union[AnnData, AnnCollection]) -> list[Tensor]:
        """Helper function to generate target ids from label mappings.

        Args:
            adata (Union[AnnData, AnnCollection]): Anndata object containing targets in `obs`.
        """
        target_ids = []
        if isinstance(self.target, str):
            targets = [self.target]
        else:
            targets = self.target

        for target in targets:
            target_mappings = self.mappings[target]['label2id']
            target_ids.append(
                torch.as_tensor(
                    adata.obs[self.target].astype('category').cat.
                    rename_categories(target_mappings).astype('int64').values))

        return target_ids

    def __call__(self, adata):
        """Returns a Torch DataLoader object."""
        return AnnLoader(adata,
                         batch_size=self.batch_size,
                         collate_fn=lambda batch: self.collate_fn(batch))

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params for model_config."""
        return dict()


def build_dataloader(dataloader_config: dict,
                     adata: Union[AnnData, AnnCollection], target: str,
                     mappings: dict) -> tuple[DataLoaderBase, dict]:
    """Builder object to get DataLoader, updated dataloader_config.

    Args:
        dataloader_config (dict): Config to build dataloader.
        adata (Union[AnnData, AnnCollection]): Data to load.
        target (str): Target column in data.obs.
        mappings (dict): Mappings of column labels to ids.
    """
    if not dataloader_config.get('name'):
        raise ValueError('DataLoader name is required!')

    dataloader_config['params'] = dataloader_config.get('params',
                                                        dict(batch_size=1))
    dataloader_config['params']['target'] = target
    dataloader_config['params']['mappings'] = mappings

    dataloader_object, dataloader_config = build_object(scalr.nn.dataloader,
                                                        dataloader_config)
    dataloader_config['params'].pop('target')
    dataloader_config['params'].pop('mappings')

    return dataloader_object(adata), dataloader_config
