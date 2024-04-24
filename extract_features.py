import os
import sys
import argparse
import numpy as np
from scp.utils import load_config, read_data, read_yaml, dump_yaml, dump_json, write_data
from scp.feature_selection import feature_chunking
from sklearn.linear_model import LogisticRegression, SGDClassifier

def main():
    # Parser to take in config file path and logging [enabled, disabled]
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', type=str, help='config.yml file')
    parser.add_argument('-l','--log', action='store_true', help='Store train-logs')

    args = parser.parse_args()
    
    # load config file
    config = load_config(args.config)
    filepath = config['filepath']
    exp_name = config['exp_name']
    exp_run = config['exp_run']

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
    if args.log:
        sys.stdout = open(f'{filepath}/feature_extraction.log','w')
    
    # loading data
    train_data = read_data(train_datapath)

    config_fs = config['feature_selection']
    if config_fs['model']['name'] == 'logistic_classifier':
        model = LogisticRegression(**config_fs['model']['params'])
    elif config_fs['model']['name'] == 'sgd_classifier':
        model = SGDClassifier(**config_fs['model']['params'])
    else:
        raise ValueError(
            'Please recheck model inside feature selection... it can be only two choices - logistic_classifier or sgd_classifier...'
        )

    chunksize = config_fs['chunksize']
    k = config_fs['top_features_stats']['k']
    aggregation_strategy = config_fs['top_features_stats']['aggregation_strategy']

    feature_chunking(train_data, target, model, chunksize, k, aggregation_strategy, filepath=f'{filepath}/feature_selection')

if __name__ == '__main__':
    main()






















