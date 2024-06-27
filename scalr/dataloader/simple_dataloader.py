from typing import Union

import torch
import anndata as ad
from anndata.experimental import AnnLoader, AnnCollection
from anndata import AnnData


def simple_dataloader(adata: Union[AnnData, AnnCollection],
                      target: str,
                      batch_size: int = 1,
                      label_mappings: dict = None):
    """
    A simple data loader to prepare inputs to be feed into linear model and corresponding labels

    Args:
        adata: anndata object containing the data
        target: corresponding metadata name to be treated as training objective in classification.
                must be present as a column_name in adata.obs
        batch_size: size of batches returned
        label_mappings: mapping the target name to respective ids

    Return:
        PyTorch DataLoader object with (X: Tensor [batch_size, features], y: Tensor [batch_size, ])
    """

    if label_mappings is None:
        label_mappings = adata.obs[target].astype(
            'category').cat.categories.tolist()
        label_mappings = {
            label_mappings[i]: i
            for i in range(len(label_mappings))
        }
    else:
        label_mappings = label_mappings[target]['label2id']

    def collate_fn(batch, target, label_mappings):
        x = batch.X.float()
        y = torch.as_tensor(
            batch.obs[target].astype('category').cat.rename_categories(
                label_mappings).astype('int64').values)
        return x, y

    return AnnLoader(
        adata,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, target, label_mappings))
