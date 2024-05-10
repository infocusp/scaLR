import anndata as ad
from anndata.experimental import AnnCollection
import yaml
import json
import os

# TODO: Merge all into single function for read and dump
def read_yaml(filepath):
    """This function returns config file loaded from yaml."""
    with open(filepath, 'r') as fh:
        config = yaml.safe_load(fh)
    return config

def read_json(filepath):
    """This function returns json file object"""
    with open(filepath, 'r') as fh:
        config = json.load(fh)
    return config

def dump_yaml(config, filepath):
    """This function stores config file to filepath"""
    with open(filepath, 'w') as fh:
        config = yaml.dump(config, fh)
    return

def dump_json(config, filepath):
    """This function stores json file to filepath"""
    with open(filepath, 'w') as fh:
        config = json.dump(config, fh, indent=2)
    return
    
def read_data(filepath, backed='r'):
    """This function reads the anndata in backed `r mode."""
    if filepath.endswith('.h5ad'):
        data = ad.read_h5ad(filepath, backed=backed)
    else:
        datas = []
        for i in range(100):
            try:
                datas.append(ad.read_h5ad(f'{filepath}/{i}.h5ad', backed=backed))
            except:
                break
        data = AnnCollection(datas)
    return data

def write_data(adata, filepath, chunksize=None):
    """This function writes the anndata."""
    if chunksize is None:
        adata.write(filepath, compression="gzip")
