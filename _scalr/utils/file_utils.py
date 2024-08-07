import os
from os import path
import json
from typing import Union

import numpy as np
import yaml
import anndata as ad
from anndata import AnnData
from anndata.experimental import AnnCollection


def read_data(filepath: str,
              backed: str = 'r') -> Union[dict, AnnData, AnnCollection]:
    """Reads a json, yaml or AnnData object file if filepath contains it.
    Returns an AnnCollection in case of a directory with chunked anndatas. 

    Args:
        filepath (str): path to `json`, `yaml` or `h5ad` file. 
                        Or directory containing multiple `h5ad` files.
        backed (str, optional): To load AnnData / AnnCollection in backed mode. Defaults to 'r'.

    Raises:
        ValueError: In case of wrong filepath provided

    Returns:
        Union[dict, AnnData, AnnCollection]
    """
    if filepath.endswith('.json'):
        data = read_json(filepath)
    elif filepath.endswith('.yaml'):
        data = read_yaml(filepath)
    elif filepath.endswith('.h5ad'):
        data = read_anndata(filepath, backed=backed)
    elif path.exists(path.join(filepath, '0.h5ad')):
        data = read_chunked_anndatas(filepath, backed=backed)
    else:
        raise ValueError('''`filepath` is not a `json`, `yaml`, or `h5ad` file,
            or directory containing `h5ad` files.
            ''')
    return data


def write_data(data: Union[dict, AnnData], filepath: str):
    """Writes data to `json`, `yaml` or `h5ad` file"""
    if filepath.endswith('.json'):
        dump_json(data, filepath)
    elif filepath.endswith('.yaml'):
        dump_yaml(data, filepath)
    elif filepath.endswith('.h5ad'):
        assert type(
            data
        ) == AnnData, 'Only AnnData objects can be written as `h5ad` files'
        dump_anndata(data, filepath)
    else:
        raise ValueError(
            '`filepath` does not contain `json`, `yaml`, or `h5ad` file')


def write_chunkwise_data(datapath: str,
                         sample_chunksize: int,
                         dirpath: str,
                         sample_inds: Union[list[int], int] = -1,
                         feature_inds: Union[list[int], int] = -1,
                         transform: function = None):
    """Write data subsets iteratively in a chunkwise manner, to ensure
    only at most `sample_chunksize` samples are loaded at a time.
    
    This function can also applies transformation on each chunk.

    Args:
        datapath (str): path/to/data to be written in chunks
        sample_chunksize (int): number of samples to be loaded at a time
        dirpath (str): path/to/directory to write the chunks of data
        sample_inds (Union[list[int], int], optional): To be used in case of chunking 
                                                       only a subset of samples. 
                                                       Defaults to all samples.
        feature_inds (Union[list[int], int], optional): To be used in case of writing
                                                        only a subset of features. 
                                                        Defaults to all features.
        transform (function): a function to apply transformation on chunked numpy array
    """
    if path.exists(dirpath): os.makedirs(dirpath)

    data = read_data(datapath)

    if sample_inds == -1: sample_inds = list(range(len(data)))
    if feature_inds == -1: feature_inds = list(range(len(data.var_names)))

    # Hacky fix for an AnnCollection working/bug
    if sample_chunksize >= len(data): sample_chunksize = len(data) - 1

    for i, (start) in enumerate(range(0, len(sample_inds), sample_chunksize)):
        data = read_data(datapath)[sample_inds]
        data = data[start:start + sample_chunksize, feature_inds]
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
    if data_config.get('feature_subset_datapaths'):
        train_data = read_data(
            data_config['feature_subset_datapaths']['train'])
        val_data = read_data(data_config['feature_subset_datapaths']['val'])
    elif data_config.get('final_datapaths'):
        train_data = read_data(data_config['final_datapaths']['train'])
        val_data = read_data(data_config['final_datapaths']['val'])
    elif data_config.get('split_datapaths'):
        train_data = read_data(data_config['split_datapaths']['train'])
        val_data = read_data(data_config['split_datapaths']['val'])
    else:
        raise ValueError('Split Datapaths not given')

    return train_data, val_data


def load_test_data_from_config(data_config):
    if data_config.get('feature_subset_datapaths'):
        test_data = read_data(data_config['feature_subset_datapaths']['test'])
    elif data_config.get('final_datapaths'):

        test_data = read_data(data_config['final_datapaths']['test'])
    elif data_config.get('split_datapaths'):

        test_data = read_data(data_config['split_datapaths']['test'])
    else:
        raise ValueError('Split Datapaths not given')

    return test_data


# Readers
def read_yaml(filepath: str) -> dict:
    """Returns the config file loaded from yaml."""
    with open(filepath, 'r') as fh:
        config = yaml.safe_load(fh)
    return config


def read_json(filepath: str) -> dict:
    """Returns the json file object"""
    with open(filepath, 'r') as fh:
        config = json.load(fh)
    return config


def read_anndata(filepath: str, backed: str = 'r') -> AnnData:
    """Returns the Anndata object from filepath"""
    data = ad.read_h5ad(filepath, backed=backed)
    return data


def read_chunked_anndatas(dirpath: str, backed: str = 'r') -> AnnCollection:
    """Returns an AnnCollection object from multiple anndatas 
    in dirpath directory
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
    """Stores the json file to filepath"""
    with open(filepath, 'w') as fh:
        config = json.dump(config, fh, indent=2)
    return


def dump_yaml(config: dict, filepath: str):
    """Stores the config file to filepath"""
    with open(filepath, 'w') as fh:
        config = yaml.dump(config, fh)
    return


def dump_anndata(adata: AnnData, filepath: str):
    """Writes the AnnData to filepath."""
    adata.write(filepath, compression="gzip")
