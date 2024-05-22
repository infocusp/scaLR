import os
import json
from typing import Union

import yaml
import anndata as ad
from anndata import AnnData
from anndata.experimental import AnnCollection


# TODO: Merge all into single function for read and dump
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


def dump_yaml(config: dict, filepath: str):
    """Stores the config file to filepath"""
    with open(filepath, 'w') as fh:
        config = yaml.dump(config, fh)
    return


def dump_json(config: dict, filepath: str):
    """Stores the json file to filepath"""
    with open(filepath, 'w') as fh:
        config = json.dump(config, fh, indent=2)
    return


def read_data(filepath: str,
              backed: str = 'r') -> Union[AnnData, AnnCollection]:
    """Reads the AnnData or an AnnCollection from a directory of chunked AnnDatas
        TODO: use glob
    """
    if filepath.endswith('.h5ad'):
        data = ad.read_h5ad(filepath, backed=backed)
    else:
        datas = []
        for i in range(len(os.listdir(filepath))):
            if os.path.exists(f'{filepath}/{i}.h5ad'):
                datas.append(
                    ad.read_h5ad(f'{filepath}/{i}.h5ad', backed=backed))
            else:
                break
        data = AnnCollection(datas)
    return data


def write_data(adata: AnnData, filepath: str, chunksize: int = None):
    """Writes the AnnData to filepath."""
    if chunksize is None:
        adata.write(filepath, compression="gzip")
    else:
        raise NotImplementedError(
            'Only `chunksize=None` available as options!')
    # TODO:
    # implement chunkwise writing of anndata to handle chunksize not None
