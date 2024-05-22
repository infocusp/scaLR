import os
import sys
import argparse

import torch
from torch import nn
import numpy as np

from scalr.utils import load_config, read_data, read_yaml, dump_yaml, dump_json
from scalr.data import generate_train_val_test_split, normalize_data, split_data


def ingest_data(config, log=True):

    dirpath = config['dirpath']
    exp_name = config['exp_name']
    exp_run = config['exp_run']

    dirpath = f'{dirpath}/{exp_name}_{exp_run}'

    data_config = config['data']
    target = data_config['target']
    normalize = data_config['normalize_data']

    process_fn = normalize_data if normalize else None

    os.makedirs(f'{dirpath}/', exist_ok=True)

    # TODO: add absl.logging functionality
    if log:
        sys.stdout = open(f'{dirpath}/data_ingestion.log', 'w')

    # Train Val Test Split
    if 'split_data' in data_config:
        os.makedirs(f'{dirpath}/data/', exist_ok=True)

        full_datapath = data_config['full_datapath']
        chunksize = data_config['chunksize']
        split_config = data_config['split_data']
        split_ratio = split_config['split_ratio']
        stratify = split_config.get('stratify', None)

        generate_train_val_test_split(full_datapath, split_ratio, target,
                                      stratify, f'{dirpath}/data', chunksize,
                                      process_fn)

    # Normalize existing splits
    if normalize and 'split_data' not in data_config:
        os.makedirs(f'{dirpath}/data/', exist_ok=True)

        chunksize = data_config['chunksize']

        for split_name in ['train', 'val', 'test']:
            split_data(data_config[f'{split_name}_datapath'], {split_name: -1},
                       f'{dirpath}/data/', chunksize, process_fn)

    # changing dirpath in config
    if normalize or 'split_data' in data_config:
        config['data']['train_datapath'] = f'{dirpath}/data/train'
        config['data']['val_datapath'] = f'{dirpath}/data/val'
        config['data']['test_datapath'] = f'{dirpath}/data/test'

        if chunksize is None:
            config['data']['train_datapath'] += '.h5ad'
            config['data']['val_datapath'] += '.h5ad'
            config['data']['test_datapath'] += '.h5ad'

    dump_yaml(config, f'{dirpath}/config.yml')
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='config.yml file')
    parser.add_argument('-l',
                        '--log',
                        action='store_true',
                        help='Store data ingestion logs')

    args = parser.parse_args()

    config = load_config(args.config)

    ingest_data(config, args.log)
