import os
import sys
import argparse

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.linear_model import LogisticRegression, SGDClassifier

from scp.utils import load_config, read_data, read_yaml, dump_yaml, dump_json, write_data
from scp.feature_selection import feature_chunking, extract_top_k_features
from scp.model import LinearModel


def extract_features(config, log=True):

    dirpath = config['dirpath']
    exp_name = config['exp_name']
    exp_run = config['exp_run']
    device = config['device']

    dirpath = f'{dirpath}/{exp_name}_{exp_run}'
    os.makedirs(f'{dirpath}/feature_selection', exist_ok=True)

    data_config = config['data']
    target = data_config['target']
    train_datapath = data_config['train_datapath']
    val_datapath = data_config['val_datapath']
    test_datapath = data_config['test_datapath']

    # TODO: add absl.logging functionality
    if log:
        sys.stdout = open(
            f'{dirpath}/feature_selection/feature_extraction.log', 'w')

    train_data = read_data(train_datapath)
    val_data = read_data(val_datapath)
    test_data = read_data(test_datapath)

    config_fs = config['feature_selection']
    weight_matrix=config_fs.get('weight_matrix', None)
    k = config_fs['top_features_stats']['k']
    aggregation_strategy = config_fs['top_features_stats'][
        'aggregation_strategy']

    if weight_matrix is None:
        chunksize = config_fs['chunksize']
        model_config = config_fs['model']
    
        feature_class_weights = feature_chunking(
            train_data,
            val_data,
            target,
            model_config,
            chunksize,
            dirpath=f'{dirpath}/feature_selection',
            device=device)
    else:
        feature_class_weights = pd.read_csv(weight_matrix, index_col=0)

    top_features = extract_top_k_features(feature_class_weights, 
                                          k, aggregation_strategy,
                                          dirpath=f'{dirpath}/feature_selection')
    
    top_features_indices = sorted([
        train_data.var_names.tolist().index(feature)
        for feature in top_features
    ])

    # Storing the Data
    if config_fs['store_on_disk']:
        if data_config['chunksize'] is None:
            train_data[:, top_features_indices].write(
                f'{dirpath}/feature_selection/train.h5ad', compression='gzip')
            val_data[:, top_features_indices].write(
                f'{dirpath}/feature_selection/val.h5ad', compression='gzip')
            test_data[:, top_features_indices].write(
                f'{dirpath}/feature_selection/test.h5ad', compression='gzip')
        else:
            chunksize = data_config['chunksize']
            os.makedirs(f'{dirpath}/feature_selection/train/', exist_ok=True)
            for i, (start) in enumerate(range(0, len(train_data), chunksize)):
                train_data = read_data(train_datapath)
                train_data = train_data[start:start + chunksize,
                                        top_features_indices]
                if not isinstance(train_data, AnnData):
                    train_data = train_data.to_adata()
                write_data(train_data,
                           f'{dirpath}/feature_selection/train/{i}.h5ad')

            os.makedirs(f'{dirpath}/feature_selection/val/', exist_ok=True)
            for i, (start) in enumerate(range(0, len(val_data), chunksize)):
                val_data = read_data(val_datapath)
                val_data = val_data[start:start + chunksize,
                                    top_features_indices]
                if not isinstance(val_data, AnnData):
                    val_data = val_data.to_adata()
                write_data(val_data,
                           f'{dirpath}/feature_selection/val/{i}.h5ad')

            os.makedirs(f'{dirpath}/feature_selection/test/', exist_ok=True)
            for i, (start) in enumerate(range(0, len(test_data), chunksize)):
                test_data = read_data(test_datapath)
                test_data = test_data[start:start + chunksize,
                                      top_features_indices]
                if not isinstance(test_data, AnnData):
                    test_data = test_data.to_adata()
                write_data(test_data,
                           f'{dirpath}/feature_selection/test/{i}.h5ad')

        config['data'][
            'train_datapath'] = f'{dirpath}/feature_selection/train'
        config['data']['val_datapath'] = f'{dirpath}/feature_selection/val'
        config['data']['test_datapath'] = f'{dirpath}/feature_selection/test'

        if data_config['chunksize'] is None:
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
                        help='Store extraction-logs')

    args = parser.parse_args()

    config = load_config(args.config)

    extract_features(config, args.log)
