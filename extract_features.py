import os
import sys
import argparse
import numpy as np
from anndata import AnnData
from scp.utils import load_config, read_data, read_yaml, dump_yaml, dump_json, write_data
from scp.feature_selection import skl_feature_chunking, nn_feature_chunking
from scp.model import LinearModel
from sklearn.linear_model import LogisticRegression, SGDClassifier

def extract_features(config, log=True):

    filepath = config['filepath']
    exp_name = config['exp_name']
    exp_run = config['exp_run']
    device = config['device']

    # create experiment directory
    filepath = f'{filepath}/{exp_name}_{exp_run}'
    os.makedirs(f'{filepath}/feature_selection', exist_ok=True)

    # Data
    data_config = config['data']
    target = data_config['target']
    train_datapath = data_config['train_datapath']
    val_datapath = data_config['val_datapath']
    test_datapath = data_config['test_datapath']

    # logging
    if log:
        sys.stdout = open(f'{filepath}/feature_extraction.log','w')
    
    # loading data
    train_data = read_data(train_datapath)
    val_data = read_data(val_datapath)
    test_data = read_data(test_datapath)

    # Feature selection configs
    config_fs = config['feature_selection']
    chunksize = config_fs['chunksize']
    k = config_fs['top_features_stats']['k']
    aggregation_strategy = config_fs['top_features_stats']['aggregation_strategy']
    model_config = config_fs['model']

    if model_config['name'] == 'nn':
        top_features = nn_feature_chunking(train_data, val_data, target, model_config, chunksize, k, aggregation_strategy, dirpath=f'{filepath}/feature_selection', device=device)
    else:
        top_features = skl_feature_chunking(train_data, target, model_config, chunksize, k, aggregation_strategy, dirpath=f'{filepath}/feature_selection')

    # Extract features to make a subset
    
    top_features_indices = sorted([train_data.var_names.tolist().index(feature) for feature in top_features])

    if config_fs['store_on_disk']:
        if data_config['chunksize'] is None:
            train_data[:,top_features_indices].write(f'{filepath}/feature_selection/train.h5ad', compression='gzip')
            val_data[:,top_features_indices].write(f'{filepath}/feature_selection/val.h5ad', compression='gzip')
            test_data[:,top_features_indices].write(f'{filepath}/feature_selection/test.h5ad', compression='gzip')
        else:
            chunksize = data_config['chunksize']
            os.makedirs(f'{filepath}/feature_selection/train/', exist_ok=True)
            for i, (start) in enumerate(range(0, len(train_data), chunksize)):
                train_data = read_data(train_datapath)
                train_data = train_data[start:start+chunksize,top_features_indices]
                if not isinstance(train_data, AnnData):
                    train_data = train_data.to_adata()
                write_data(train_data, f'{filepath}/feature_selection/train/{i}.h5ad')

            os.makedirs(f'{filepath}/feature_selection/val/', exist_ok=True)
            for i, (start) in enumerate(range(0, len(val_data), chunksize)):
                val_data = read_data(val_datapath)
                val_data = val_data[start:start+chunksize,top_features_indices]
                if not isinstance(val_data, AnnData):
                    val_data = val_data.to_adata()
                write_data(val_data, f'{filepath}/feature_selection/val/{i}.h5ad')

            os.makedirs(f'{filepath}/feature_selection/test/', exist_ok=True)
            for i, (start) in enumerate(range(0, len(test_data), chunksize)):
                test_data = read_data(test_datapath)
                test_data = test_data[start:start+chunksize,top_features_indices]
                if not isinstance(test_data, AnnData):
                    test_data = test_data.to_adata()
                write_data(test_data, f'{filepath}/feature_selection/test/{i}.h5ad')

        config['data']['train_datapath'] = f'{filepath}/feature_selection/train'
        config['data']['val_datapath'] = f'{filepath}/feature_selection/val'
        config['data']['test_datapath'] = f'{filepath}/feature_selection/test'

        if chunksize is None:
            config['data']['train_datapath'] += '.h5ad'
            config['data']['val_datapath'] += '.h5ad'
            config['data']['test_datapath'] += '.h5ad'

    dump_yaml(config, f'{filepath}/config.yml')
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', type=str, help='config.yml file')
    parser.add_argument('-l','--log', action='store_true', help='Store extraction-logs')

    args = parser.parse_args()
    
    # load config file
    config = load_config(args.config)

    extract_features(config, args.log)



















