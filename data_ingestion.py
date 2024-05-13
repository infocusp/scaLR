import os
import sys
import argparse
import torch
from torch import nn
import numpy as np
from scp.utils import load_config, read_data, read_yaml, dump_yaml, dump_json, split_data, generate_split
from scp import Trainer

def data_ingestion(config, log=True):

    filepath = config['filepath']
    exp_name = config['exp_name']
    exp_run = config['exp_run']

    # create experiment directory
    filepath = f'{filepath}/{exp_name}_{exp_run}'
    
    data_config = config['data']
    target = data_config['target']

    os.makedirs(f'{filepath}/', exist_ok=True)
    
    if log:
        sys.stdout = open(f'{filepath}/data_ingestion.log','w')

    # Normalization, Harmonization, Batch Correction to be added here
    
    # Splitting the data
    if 'split_data' in data_config:
        os.makedirs(f'{filepath}/data/', exist_ok=True)
        
        split_config = data_config['split_data']
        full_datapath = split_config['full_datapath']
        split_ratio = split_config['split_ratio']
        chunksize = data_config['chunksize']
        stratify = split_config.get('stratify', None)

        # Generate split indices
        data_split = generate_split(full_datapath, split_ratio, target, stratify, f'{filepath}/data/data_split.json')

        # Split the data
        split_data(full_datapath, data_split, dirpath=f'{filepath}/data', chunksize=chunksize)

        config['data']['train_datapath'] = f'{filepath}/data/train'
        config['data']['val_datapath'] = f'{filepath}/data/val'
        config['data']['test_datapath'] = f'{filepath}/data/test'

        if chunksize is None:
            config['data']['train_datapath'] += '.h5ad'
            config['data']['val_datapath'] += '.h5ad'
            config['data']['test_datapath'] += '.h5ad'

    dump_yaml(config, f'{filepath}/config.yml')
    return config
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', type=str, help='config.yml file')
    parser.add_argument('-l','--log', action='store_true', help='Store data ingestion logs')

    args = parser.parse_args()
    
    # load config file
    config = load_config(args.config)

    data_ingestion(config, args.log)






















