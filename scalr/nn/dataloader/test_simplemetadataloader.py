'''This is a test file for simplemetadataloader.'''

import anndata
import numpy as np
import pandas as pd

from scalr.nn.dataloader import build_dataloader


def test_metadataloader():

    # Setting seed for reproducibility
    np.random.seed(0)

    # Creating anndata object.
    adata = anndata.AnnData(X=np.random.rand(15, 7))
    adata.obs = pd.DataFrame.from_dict({
        'celltype': np.random.choice(['B', 'C', 'DC', 'T'], size=15),
        'batch': np.random.choice(['batch1', 'batch2'], size=15),
        'env': np.random.choice(['env1', 'env2', 'env3'], size=15)
    })
    adata.obs.index = adata.obs.index.astype('O')

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
