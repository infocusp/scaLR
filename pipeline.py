import os
import sys
import argparse

from scalr.utils import load_config
from train import train
from evaluate import evaluate
from feature_extraction import extract_features
from data_ingestion import ingest_data


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
