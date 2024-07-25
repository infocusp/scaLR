import os
from os import path
import json
from typing import Callable

import anndata as ad
from anndata import AnnData
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from ..utils import write_data, dump_json, read_data


def _generate_train_val_test_split_indices(datapath: str,
                                           split_ratio: list[float],
                                           target: str,
                                           stratify: str = None,
                                           dirpath: str = None) -> dict:
    """Generate a list of indices for train/val/test split of whole dataset

    Args:
        datapath: path to full data
        split_ratio: ratio to split number of samples in
        target: target for classification present in `obs`.
        stratify: optional parameter to stratify the split upon parameter.
        dirpath: dirpath to store generated split in json format

    Returns:
        dict with 'train', 'test' and 'val' indices list.

    """

    adata = read_data(datapath)
    metadata = adata.obs
    metadata['inds'] = range(len(metadata))
    n_cls = metadata[target].nunique()

    total_ratio = sum(split_ratio)
    train_ratio = split_ratio[0] / total_ratio
    val_ratio = split_ratio[1] / total_ratio
    test_ratio = split_ratio[2] / total_ratio

    Splitter = GroupShuffleSplit if stratify else StratifiedShuffleSplit

    test_splitter = Splitter(test_size=test_ratio,
                             n_splits=10,
                             random_state=42)
    training_inds, testing_inds = next(
        test_splitter.split(metadata,
                            metadata[target],
                            groups=metadata[stratify] if stratify else None))

    if len(metadata[target].iloc[testing_inds].unique()) != n_cls:
        print('WARNING: All classes are not present in Test set')

    train_data = metadata.iloc[training_inds]

    val_splitter = Splitter(test_size=val_ratio / (val_ratio + train_ratio),
                            n_splits=10,
                            random_state=42)
    fake_train_inds, fake_val_inds = next(
        val_splitter.split(train_data,
                           train_data[target],
                           groups=train_data[stratify] if stratify else None))

    true_test_inds = testing_inds.tolist()
    true_val_inds = train_data.iloc[fake_val_inds]['inds'].tolist()
    true_train_inds = train_data.iloc[fake_train_inds]['inds'].tolist()

    if len(metadata[target].iloc[true_val_inds].unique()) != n_cls:
        print('WARNING: All classes are not present in Validation set')

    if len(metadata[target].iloc[true_train_inds].unique()) != n_cls:
        print('WARNING: All classes are not present in Train set')

    assert len(set(true_test_inds).intersection(true_train_inds)
               ) == 0, "Test and Train sets contain overlapping samples"
    assert len(set(true_val_inds).intersection(true_train_inds)
               ) == 0, "Validation and Train sets contain overlapping samples"
    assert len(set(true_val_inds).intersection(true_test_inds)
               ) == 0, "Test and Validation sets contain overlapping samples"

    print('Length of train set: ', len(true_train_inds))
    print('Length of val set: ', len(true_val_inds))
    print('Length of test set: ', len(true_test_inds), flush=True)

    data_split = {
        'train': true_train_inds,
        'val': true_val_inds,
        'test': true_test_inds
    }

    if dirpath is not None:
        dump_json(data_split, path.join(dirpath, 'data_split.json'))

    return data_split


def split_data(datapath: str,
               data_split: dict,
               dirpath: str,
               sample_chunksize: int = None,
               process_fn: Callable = None,
               **kwargs):
    """Split the full data based upon generated indices lists and write it to disk.

    Args:
        datapath: path to full dataset
        data_split: dict containing list of indices for each split, `-1` as value indicates all indices
        dirpath: path to store new split data.
        sample_chunksize: numberadata of samples to store in one chunk, after splitting the data.
        process_fn: a function which takes in data chunk to perform operations on it like Normalization
        **kwargs: keyword arguments to pass to `process_fn` apart from adata's numpy array.
    """
    total_samples = len(read_data(datapath))

    for split_name in data_split.keys():
        if data_split[split_name] == -1:
            data_split[split_name] = list(range(total_samples))
        if not sample_chunksize:
            adata = read_data(datapath).to_memory()
            if process_fn is not None:
                print('\nNormalizing the data...')
                if not isinstance(adata.X, np.ndarray):
                    adata.X = adata.X.A
                adata.X = process_fn(adata.X, **kwargs)
            write_data(adata[data_split[split_name]],
                       path.join(dirpath, f'{split_name}.h5ad'))
        else:
            os.makedirs(path.join(dirpath, split_name), exist_ok=True)
            curr_chunksize = len(
                data_split[split_name]) - 1 if sample_chunksize >= len(
                    data_split[split_name]) else sample_chunksize
            for i, (start) in enumerate(
                    range(0, len(data_split[split_name]), curr_chunksize)):
                adata = read_data(datapath)
                adata = adata[data_split[split_name][start:start +
                                                     curr_chunksize]]
                if not isinstance(adata, AnnData):
                    adata = adata.to_adata()
                adata = adata.to_memory()
                if process_fn is not None:
                    print('\nNormalizing the data...')
                    if not isinstance(adata.X, np.ndarray):
                        adata.X = adata.X.A
                    adata.X = process_fn(adata.X, **kwargs)
                write_data(adata, path.join(dirpath, split_name, f'{i}.h5ad'))


def generate_train_val_test_split(datapath: str,
                                  split_ratio: list[float],
                                  target: str,
                                  stratify: str = None,
                                  dirpath: str = None,
                                  sample_chunksize: int = None,
                                  process_fn: Callable = None,
                                  **kwargs):
    """Generate a list of indices for train/val/test split of whole dataset and writes new data splits
    to disk.

    Args:
        datapath: path to full data
        split_ratio: ratio to split number of samples in
        target: target for classification present in `obs`.
        stratify: optional parameter to stratify the split upon parameter.
        dirpath: dirpath to store generated split in json format
        sample_chunksize: number of samples to store in one chunk, after splitting the data.

    Returns:
        dict with 'train', 'test' and 'val' indices list.
    """

    data_split = _generate_train_val_test_split_indices(
        datapath, split_ratio, target, stratify, dirpath)

    split_data(datapath, data_split, dirpath, sample_chunksize, process_fn,
               **kwargs)
