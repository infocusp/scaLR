"""This is a test file for simpledataloader."""

from scalr.nn.dataloader import build_dataloader
from scalr.utils import generate_dummy_anndata


def test_metadataloader():
    """This function tests features shape returned by simpledataloader for the below 2 cases.
        1. #features are consistent with feature_subsetsize. No padding is required.
        2. #features are less than feature_subsetsize. This case needs padding.
    """

    # Generating dummy anndata.
    n_samples = 30
    n_features = 13
    adata = generate_dummy_anndata(n_samples=n_samples, n_features=n_features)

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

    #                         Test case 1
    # Expected features shape after dataloading is (batch_size, 13).
    # So no padding is required as adata n_features=13. But we can pass
    # `padding=feature_subsetsize` in dataloader_config.

    ## Defining required parameters for simpledataloader.
    feature_subsetsize = 13
    dataloader_config = {
        'name': 'SimpleDataLoader',
        'params': {
            'batch_size': 10,
            'padding': feature_subsetsize,
        }
    }
    dataloader, _ = build_dataloader(dataloader_config=dataloader_config,
                                     adata=adata,
                                     target='celltype',
                                     mappings=mappings)

    ## Comparing expecting features shape after using metadatloader.
    for feature, _ in dataloader:
        assert feature.shape[
            1] == feature_subsetsize, f"There is some issue in simpledataloader."\
        f" Expected #features({n_features}) != Observed #features({feature.shape[1]}). Please check!"
        # Breaking, as checking only the first batch is enough.
        break

    #                         Test case 2
    # Expected features shape after dataloading is (batch_size, 20).
    # So padding is required as adata n_features=13. Hence 7 columns should be padded in dataloader.

    ## Defining required parameters for simpledataloader.
    feature_subsetsize = 20
    dataloader_config = {
        'name': 'SimpleDataLoader',
        'params': {
            'batch_size': 10,
            'padding': feature_subsetsize,
        }
    }
    dataloader, _ = build_dataloader(dataloader_config=dataloader_config,
                                     adata=adata,
                                     target='celltype',
                                     mappings=mappings)

    ## Comparing expected features shape after using metadatloader.
    for feature, _ in dataloader:
        assert feature.shape[
            1] == feature_subsetsize, f"There is some issue in simpledataloader."\
        f" Expected #features({feature_subsetsize}) != Observed #features({feature.shape[1]}). Please check!"
        # Breaking, as checking only the first batch is enough.
        break
