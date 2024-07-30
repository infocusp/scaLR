from typing import Union

import numpy as np
import torch
import anndata as ad
from anndata.experimental import AnnLoader, AnnCollection
from anndata import AnnData


def simple_dataloader(adata: Union[AnnData, AnnCollection],
                      target: str,
                      batch_size: int = 1,
                      label_mappings: dict = None,
                      batch_mappings: dict = None,
                      shuffle: bool = False):
    """
    A simple data loader to prepare inputs to be feed into linear model and corresponding labels

    Args:
        adata: anndata object containing the data
        target: corresponding metadata name to be treated as training objective in classification.
                must be present as a column_name in adata.obs
        batch_size: size of batches returned
        label_mappings: mapping the target name to respective ids
        shuffle: Shuffle indices before sampling batch.
        batch_mappings: mapping of batches to respective ids

    Return:
        PyTorch DataLoader object with (X: Tensor [batch_size, features], y: Tensor [batch_size, ])
    """

    label_mappings = label_mappings[target]['label2id']

    def collate_fn(batch, target, label_mappings, batch_mappings):
        x = batch.X.float()
        if batch_mappings:
            x = torch.cat(
                (x,
                 torch.as_tensor(batch.obs['batch'].astype(
                     'category').cat.rename_categories(batch_mappings).astype(
                         'int64').values).reshape(-1, 1)),
                dim=1)
        y = torch.as_tensor(
            batch.obs[target].astype('category').cat.rename_categories(
                label_mappings).astype('int64').values)
        return x, y

    return AnnLoader(adata,
                     batch_size=batch_size,
                     shuffle=shuffle,
                     collate_fn=lambda batch: collate_fn(
                         batch, target, label_mappings, batch_mappings))
