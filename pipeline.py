import os
import sys
import argparse
from scp.utils import load_config
from train import train
from evaluate import evaluate
from extract_features import extract_features
from data_ingestion import data_ingestion

def main():
    # Parser to take in config file path and logging [enabled, disabled]
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', type=str, help='config.yml file path')
    parser.add_argument('-l','--log', action='store_true', help='flag to store logs for the experiment')

    args = parser.parse_args()
    
    # load config file
    config = load_config(args.config)
    log = args.log
    
    config = data_ingestion(config, log)

    if 'feature_selection' in config:
        config = extract_features(config, log)

    if 'training' in config:
        config = train(config, log)

    if 'evaluation' in config:
        config = evaluate(config, log)

if __name__ == '__main__':
    main()