import argparse
import os
import random
import sys

import numpy as np
import torch

from config.utils import load_config
from data_ingestion import ingest_data
from evaluate import evaluate
from feature_extraction import extract_features
from train import train


def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        help='config.yml file path',
                        required=True)
    parser.add_argument('-l',
                        '--log',
                        action='store_true',
                        help='flag to store logs for the experiment')

    args = parser.parse_args()
    set_seed(42)

    config = load_config(args.config)
    log = args.log

    if config.get('data') and ('target' in config['data']):
        print('\nInitializing data ingestion...')
        config = ingest_data(config, log)

    if 'feature_selection' in config:
        print('\nInitializing feature selection...')
        config = extract_features(config, log)

    if 'training' in config:
        print('\nInitializing model training...')
        config = train(config, log)

    if 'evaluation' in config:
        print('\nInitializing model evaluation...')
        config = evaluate(config, log)


if __name__ == '__main__':
    main()
