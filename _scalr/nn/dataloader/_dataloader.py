from anndata.experimental import AnnLoader, AnnCollection
from typing import Union
from anndata import AnnData
from typing import Union
import _scalr
from _scalr.utils import build_object
import torch


class DataLoaderBase:

    def __init__(
        self,
        batch_size: int = 1,
        target: Union[str, list[str]] = None,
        mappings: dict = None,
    ):
        """
        Args:
            batch_size (int, optional): _description_. Defaults to 1.
            target ([str, list[str]]): list of target. Defaults to None.
            mappings (dict): list of label mappings of each target to . 
                              Defaults to None.
        """
        self.batch_size = batch_size
        self.target = target
        self.mappings = mappings

    def collate_fn(self, batch):
        """Given an input anndata of batch_size,
        the collate function creates inputs and outputs.
        It can also be used to perform batch-wise 
        operations.
        """
        pass

    def get_targets_ids_from_mappings(self, adata: Union[AnnData,
                                                         AnnCollection]):
        """Helper function to generate 

        Args:
            adata (Union[AnnData, AnnCollection]): anndata object containing
                                                   targets in `obs`
        """
        target_ids = []
        if isinstance(self.target, str): targets = [self.target]
        else: targets = self.target

        for target in targets:
            target_mappings = self.mappings[target]['label2id']
            target_ids.append(
                torch.as_tensor(
                    adata.obs[self.target].astype('category').cat.
                    rename_categories(target_mappings).astype('int64').values))

        return target_ids

    def __call__(self, adata):
        """Returns a Torch DataLoader object"""
        return AnnLoader(adata,
                         batch_size=self.batch_size,
                         collate_fn=lambda batch: self.collate_fn(batch))

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params for model_config"""
        return dict()


def build_dataloader(dataloader_config, adata):
    dataloader_object, dataloader_config = build_object(
        _scalr.nn.dataloader, dataloader_config)
    return dataloader_object(adata), dataloader_config
