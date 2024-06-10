import os
import sys
import argparse
from config.utils import load_config
from train import train
from evaluate import evaluate
from feature_extraction import extract_features
from data_ingestion import ingest_data
import torch
import random
import numpy as np

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
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)     

    config = load_config(args.config)
    log = args.log

    config = ingest_data(config, log)

    if 'feature_selection' in config:
        config = extract_features(config, log)

    if 'training' in config:
        config = train(config, log)

    if 'evaluation' in config:
        config = evaluate(config, log)


if __name__ == '__main__':
    main()
