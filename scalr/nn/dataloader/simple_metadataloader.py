"""This file is the implementation of simple metadataloader."""

from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch

from scalr.nn.dataloader import SimpleDataLoader


class SimpleMetaDataLoader(SimpleDataLoader):
    """Class for simple metadataloader.
    
    Simple MetaDataLoader converts all adata values to inputs, concat specified metadata columns as onehotencoded vector
    to feature data and map target columns in metadata to output labels.

    Returns:
        PyTorch DataLoader object with (X: Tensor [batch_size, features], y: Tensor [batch_size, ]).
    """

    def __init__(self,
                 batch_size: int,
                 target: str,
                 mappings: dict,
                 metadata_col: list[str],
                 padding: int = None):
        """
        Args:
            batch_size (int): Number of samples to be loaded in each batch.
            target (str): Corresponding metadata name to be treated as training
                          objective in classification. Must be present as a column_name in `adata.obs`.
            mappings (dict): Mapping the target name to respective ids.
            metadata_col (list): List of metadata columns to be onehotencoded.
            padding (int): Padding size incase of #features < model input size.
        """
        super().__init__(batch_size=batch_size,
                         target=target,
                         mappings=mappings,
                         padding=padding)
        self.mappings = mappings
        self.metadata_col = metadata_col

        # Generating OneHotEncoder object for specified metadata_col.
        self.metadata_onehotencoder = {}
        for col in self.metadata_col:
            ohe = OneHotEncoder(handle_unknown='ignore')
            ohe.fit(np.array(sorted(mappings[col]['id2label'])).reshape(-1, 1))
            self.metadata_onehotencoder[col] = ohe

    def collate_fn(
        self,
        adata_batch: Union[AnnData, AnnCollection],
    ):
        """Given an input anndata of batch_size, the collate function creates inputs and outputs.

        Args:
            adata_batch (Union[AnnData, AnnCollection]): Anndata view object with batch_size samples.

        Returns:
            Tuple(x, y): Input to model, output from data.
        """

        # Getting x & y
        x, y = super().collate_fn(adata_batch)

        # One hot encoding requested metadata columns and adding to features data.
        for col in self.metadata_col:
            x = torch.cat(
                (x,
                 torch.as_tensor(self.metadata_onehotencoder[col].transform(
                     adata_batch.obs[col].values.reshape(-1, 1)).A,
                                 dtype=torch.float32)),
                dim=1)
        return x, y

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params for model_config."""
        return dict(batch_size=1, target=None, mappings=dict())
