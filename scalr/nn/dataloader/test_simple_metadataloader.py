'''This is a test file for simplemetadataloader.'''

import anndata
import numpy as np
import pandas as pd

from scalr.nn.dataloader import build_dataloader
from scalr.utils import generate_dummy_anndata


def test_metadataloader():

    # Generating dummy anndata.
    adata = generate_dummy_anndata(n_samples=15, n_features=7)

    # Generating mappings for anndata obs columns.
    mappings = {}
    for column_name in adata.obs.columns:
        mappings[column_name] = {}

        id2label = []
        id2label += adata.obs[column_name].astype(
            'category').cat.categories.tolist()

        label2id = {id2label[i]: i for i in range(len(id2label))}
        mappings[column_name]['id2label'] = id2label
        mappings[column_name]['label2id'] = label2id

    # Defining required parameters for metadataloader.
    metadata_col = ['batch', 'env']
    dataloader_config = {
        'name': 'SimpleMetaDataLoader',
        'params': {
            'batch_size': 10,
            'metadata_col': metadata_col
        }
    }
    dataloader, _ = build_dataloader(dataloader_config=dataloader_config,
                                     adata=adata,
                                     target='celltype',
                                     mappings=mappings)

    # Comparing expecting features shape after using metadatloader.
    for feature, _ in dataloader:
        assert feature.shape[1] == len(
            adata.var_names) + adata.obs[metadata_col].nunique().sum()
        # Breaking, as checking only first batch is enough.
        break
