import anndata as ad
import yaml
import json

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
    if 'h5ad' in filepath:
        data = ad.read_h5ad(filepath, backed=backed)
    else:
        raise ValueError('Only .h5ad files supported!!!')
    return data

def write_data(data, filepath):
    """This function writes the anndata."""
    adata.write(filepath, compression="gzip")

# def make_anndata_from_csv(features_data, metadata):
#     """Returns anndata for provided features data and metadata(targets).

#     if you have csv file for both then you can create a anndata using this function.

#     """

