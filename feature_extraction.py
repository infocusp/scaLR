import os
from os import path
import sys
import argparse

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.linear_model import LogisticRegression, SGDClassifier

from config.utils import load_config
from scalr.utils import read_data, read_yaml, dump_yaml, dump_json, write_data
from scalr.feature_selection import feature_chunking, extract_top_k_features
from scalr.model import LinearModel


def extract_features(config, log=True):

    dirpath = config['dirpath']
    exp_name = config['exp_name']
    exp_run = config['exp_run']
    device = config['device']

    dirpath = path.join(dirpath, f'{exp_name}_{exp_run}')
    featurespath = path.join(dirpath, 'feature_selection')
    os.makedirs(featurespath, exist_ok=True)

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
    weight_matrix = config_fs.get('weight_matrix', None)
    k = config_fs['top_features_stats']['k']
    aggregation_strategy = config_fs['top_features_stats'][
        'aggregation_strategy']

    if weight_matrix is None:
        feature_chunksize = config_fs['feature_chunksize']
        model_config = config_fs['model']

        feature_class_weights = feature_chunking(train_data,
                                                 val_data,
                                                 target,
                                                 model_config,
                                                 feature_chunksize,
                                                 dirpath=featurespath,
                                                 device=device)
    else:
        feature_class_weights = pd.read_csv(weight_matrix, index_col=0)

    top_features = extract_top_k_features(feature_class_weights,
                                          k,
                                          aggregation_strategy,
                                          dirpath=featurespath)

    top_features_indices = sorted([
        train_data.var_names.tolist().index(feature)
        for feature in top_features
    ])

    # Storing the Data
    if config_fs['store_on_disk']:
        if data_config['sample_chunksize'] is None:
            train_data[:, top_features_indices].write(path.join(
                featurespath, 'train.h5ad'),
                                                      compression='gzip')
            val_data[:, top_features_indices].write(path.join(
                featurespath, 'val.h5ad'),
                                                    compression='gzip')
            test_data[:, top_features_indices].write(path.join(
                featurespath, 'test.h5ad'),
                                                     compression='gzip')
        else:
            sample_chunksize = data_config['sample_chunksize']
            trainpath = path.join(featurespath, 'train')
            os.makedirs(trainpath, exist_ok=True)
            for i, (start) in enumerate(
                    range(0, len(train_data), sample_chunksize)):
                train_data = read_data(train_datapath)
                train_data = train_data[start:start + sample_chunksize,
                                        top_features_indices]
                if not isinstance(train_data, AnnData):
                    train_data = train_data.to_adata()
                write_data(train_data, path.join(trainpath, f'{i}.h5ad'))

            valpath = path.join(featurespath, 'val')
            os.makedirs(valpath, exist_ok=True)
            for i, (start) in enumerate(
                    range(0, len(val_data), sample_chunksize)):
                val_data = read_data(val_datapath)
                val_data = val_data[start:start + sample_chunksize,
                                    top_features_indices]
                if not isinstance(val_data, AnnData):
                    val_data = val_data.to_adata()
                write_data(val_data, path.join(valpath, f'{i}.h5ad'))

            testpath = path.join(featurespath, 'test')
            os.makedirs(testpath, exist_ok=True)
            for i, (start) in enumerate(
                    range(0, len(test_data), sample_chunksize)):
                test_data = read_data(test_datapath)
                test_data = test_data[start:start + sample_chunksize,
                                      top_features_indices]
                if not isinstance(test_data, AnnData):
                    test_data = test_data.to_adata()
                write_data(test_data, path.join(testpath, f'{i}.h5ad'))

        config['data']['train_datapath'] = trainpath
        config['data']['val_datapath'] = valpath
        config['data']['test_datapath'] = testpath

        if data_config['sample_chunksize'] is None:
            config['data']['train_datapath'] += '.h5ad'
            config['data']['val_datapath'] += '.h5ad'
            config['data']['test_datapath'] += '.h5ad'

    dump_yaml(config, path.join(dirpath, 'config.yml'))
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
