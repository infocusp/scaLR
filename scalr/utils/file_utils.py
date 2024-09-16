"""This file contains functions related to file read-write."""

import json
import os
from os import path
from typing import Union

from anndata import AnnData
import anndata as ad
from anndata.experimental import AnnCollection
import numpy as np
import pandas as pd
import yaml

from scalr.utils.logger import FlowLogger


def read_data(filepath: str,
              backed: str = 'r',
              index_col: int = 0) -> Union[dict, AnnData, AnnCollection]:
    """This function reads a json, yaml, csv or AnnData object file if the file path contains it.
    
    Returns an AnnCollection in case of a directory with chunked anndatas.

    Args:
        filepath (str): path to `json`, `yaml` or `h5ad` file.
                        Or directory containing multiple `h5ad` files.
        backed (str, optional): To load AnnData / AnnCollection in backed mode. Defaults to 'r'.

    Raises:
        ValueError: In case of the wrong file path provided.

    Returns:
        Union[dict, AnnData, AnnCollection].
    """
    if filepath.endswith('.json'):
        data = read_json(filepath)
    elif filepath.endswith('.yaml'):
        data = read_yaml(filepath)
    elif filepath.endswith('.csv'):
        data = read_csv(filepath, index_col=index_col)
    elif filepath.endswith('.h5ad'):
        data = read_anndata(filepath, backed=backed)
    elif path.exists(path.join(filepath, '0.h5ad')):
        data = read_chunked_anndatas(filepath, backed=backed)
    else:
        raise ValueError(
            '''`filepath` is not a `json`, `yaml`, `csv` or `h5ad` file,
            or directory containing `h5ad` files.
            ''')
    return data


def write_data(data: Union[dict, AnnData, pd.DataFrame], filepath: str):
    """This function writes data to `json`, `yaml`, `csv` or `h5ad` file."""
    if filepath.endswith('.json'):
        dump_json(data, filepath)
    elif filepath.endswith('.yaml'):
        dump_yaml(data, filepath)
    elif filepath.endswith('.csv'):
        dump_csv(data, filepath)
    elif filepath.endswith('.h5ad'):
        assert type(
            data
        ) == AnnData, 'Only AnnData objects can be written as `h5ad` files'
        dump_anndata(data, filepath)
    else:
        raise ValueError(
            '`filepath` does not contain `json`, `yaml`, or `h5ad` file')


def write_chunkwise_data(full_data: Union[AnnData, AnnCollection],
                         sample_chunksize: int,
                         dirpath: str,
                         sample_inds: list[int] = None,
                         feature_inds: list[int] = None,
                         transform=None):
    """This function writes data subsets iteratively in a chunkwise manner, to ensure
    only at most `sample_chunksize` samples are loaded at a time.

    This function can also apply transformation on each chunk.

    Args:
        full_data (Union[AnnData, AnnCollection]): data to be written in chunks.
        sample_chunksize (int): number of samples to be loaded at a time.
        dirpath (str): path/to/directory to write the chunks of data.
        sample_inds (list[int], optional): To be used in case of chunking
                                                       only a subset of samples.
                                                       Defaults to all samples.
        feature_inds (list[int], optional): To be used in case of writing
                                                        only a subset of features.dataframe.
                                                        Defaults to all features.
        transform (function): a function to apply a transformation on a chunked numpy array.
    """
    if not path.exists(dirpath):
        os.makedirs(dirpath)

    if not sample_inds:
        sample_inds = list(range(len(full_data)))

    # Hacky fixes for an AnnCollection working/bug.
    if sample_chunksize >= len(sample_inds):
        sample_chunksize = len(sample_inds) - 1

    for col in full_data.obs.columns:
        full_data.obs[col] = full_data.obs[col].astype('category')

    for i, (start) in enumerate(range(0, len(sample_inds), sample_chunksize)):
        if feature_inds:
            data = full_data[sample_inds[start:start + sample_chunksize],
                             feature_inds]
        else:
            data = full_data[sample_inds[start:start + sample_chunksize]]

        if not isinstance(data, AnnData):
            data = data.to_adata()
        data = data.to_memory()

        # Transformation
        if transform:
            if not isinstance(data.X, np.ndarray):
                data.X = data.X.A
            data.X = transform(data.X)

        write_data(data, path.join(dirpath, f'{i}.h5ad'))


def load_train_val_data_from_config(data_config):
    """This function returns train & validation data from the data config.

    Args:
        data_config: Data config.
    """

    flow_logger = FlowLogger('File Utils')

    train_val_test_paths = data_config.get('train_val_test')
    if not train_val_test_paths:
        raise ValueError('Split Datapaths not given')

    if train_val_test_paths.get('feature_subset_datapaths'):
        train_data = read_data(
            path.join(train_val_test_paths['feature_subset_datapaths'],
                      'train'))
        val_data = read_data(
            path.join(train_val_test_paths['feature_subset_datapaths'], 'val'))
        flow_logger.info('Train&Val Data Loaded from Feature subset datapaths')

    elif train_val_test_paths.get('final_datapaths'):
        train_data = read_data(
            path.join(train_val_test_paths['final_datapaths'], 'train'))
        val_data = read_data(
            path.join(train_val_test_paths['final_datapaths'], 'val'))
        flow_logger.info('Train&Val Data Loaded from Final datapaths')

    elif train_val_test_paths.get('split_datapaths'):
        train_data = read_data(
            path.join(train_val_test_paths['split_datapaths'], 'train'))
        val_data = read_data(
            path.join(train_val_test_paths['split_datapaths'], 'val'))
        flow_logger.info('Train&Val Data Loaded from Split datapaths')

    else:
        raise ValueError('Split Datapaths not given')

    return train_data, val_data


def load_test_data_from_config(data_config):
    """This function returns test data from the data config.

    Args:
        data_config: Data config.
    """
    flow_logger = FlowLogger('File Utils')

    train_val_test_paths = data_config.get('train_val_test')
    if not train_val_test_paths:
        raise ValueError('Split Datapaths not given')

    if train_val_test_paths.get('feature_subset_datapaths'):
        test_data = read_data(
            path.join(train_val_test_paths['feature_subset_datapaths'], 'test'))
        flow_logger.info('Test Data Loaded from Feature subset datapaths')

    elif train_val_test_paths.get('final_datapaths'):
        test_data = read_data(
            path.join(train_val_test_paths['final_datapaths'], 'test'))
        flow_logger.info('Test Data Loaded from Final datapaths')

    elif train_val_test_paths.get('split_datapaths'):
        test_data = read_data(
            path.join(train_val_test_paths['split_datapaths'], 'test'))
        flow_logger.info('Test Data Loaded from Split datapaths')

    else:
        raise ValueError('Split Datapaths not given')

    return test_data


# Readers
def read_yaml(filepath: str) -> dict:
    """This function returns the config file loaded from yaml."""
    with open(filepath, 'r') as fh:
        config = yaml.safe_load(fh)
    return config


def read_json(filepath: str) -> dict:
    """This file returns the json file object."""
    with open(filepath, 'r') as fh:
        config = json.load(fh)
    return config


def read_csv(filepath: str, index_col: int = 0) -> pd.DataFrame:
    """This file returns the DataFrame file object."""
    return pd.read_csv(filepath, index_col=index_col)


def read_anndata(filepath: str, backed: str = 'r') -> AnnData:
    """This file returns the Anndata object from filepath."""
    data = ad.read_h5ad(filepath, backed=backed)
    return data


def read_chunked_anndatas(dirpath: str, backed: str = 'r') -> AnnCollection:
    """This file returns an AnnCollection object from multiple anndatas
    in dirpath directory.
    """
    datas = []
    for i in range(len(os.listdir(dirpath))):
        if os.path.exists(path.join(dirpath, f'{i}.h5ad')):
            datas.append(
                read_anndata(path.join(dirpath, f'{i}.h5ad'), backed=backed))
        else:
            break
    data = AnnCollection(datas)
    return data


# Writers
def dump_json(config: dict, filepath: str):
    """This function stores the json file to filepath."""
    with open(filepath, 'w') as fh:
        config = json.dump(config, fh, indent=2)
    return


def dump_yaml(config: dict, filepath: str):
    """This function stores the config file to filepath."""
    with open(filepath, 'w') as fh:
        config = yaml.dump(config, fh)
    return


def dump_csv(df: pd.DataFrame, filepath: str):
    """This function stores the config file to filepath."""
    df.to_csv(filepath)
    return


def dump_anndata(adata: AnnData, filepath: str):
    """This function writes the AnnData to filepath."""
    adata.write(filepath, compression="gzip")
