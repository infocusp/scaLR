import argparse
import os
from os import path
import random
import sys

import numpy as np
import torch

from _scalr.utils import set_seed, write_data, read_data
from _scalr.data_ingestion_pipeline import DataIngestionPipeline
# from _scalr.feature_extraction_pipeline import FeatureExtraction
from _scalr.model_training_pipeline import ModelTrainingPipeline
# from _scalr.downstream_analysis_pipeline import DownstreamAnalysis


def get_args():
    """To get the command line arguments"""
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
    return args


if __name__ == '__main__':

    set_seed(42)
    args = get_args()

    config = read_data(args.config)
    log = args.log

    dirpath = config['dirpath']
    exp_name = config['exp_name']
    exp_run = config['exp_run']
    dirpath = os.path.join(dirpath, f'{exp_name}_{exp_run}')
    device = config['device']
    os.makedirs(dirpath, exist_ok=True)

    #PIPELINE RUN

    ingest_data = DataIngestionPipeline(config['data'], dirpath)
    ingest_data.generate_train_val_test_split()
    ingest_data.preprocess_data()
    ingest_data.generate_mappings()
    config['data'] = ingest_data.get_updated_config()
    write_data(config, path.join(dirpath, 'config.yaml'))

    if config.get('final_training'):
        model_trainer = ModelTrainingPipeline(
            config['final_training']['model'],
            config['final_training']['train_config'], dirpath, device)

        model_trainer.load_data_and_targets_from_config(config['data'])
        model_trainer.build_model_training_artifacts()
        model_trainer.train()
        model_config, train_config = model_trainer.get_updated_config()
        config['final_training']['model'] = model_config
        config['final_training']['train_config'] = train_config
        write_data(config, path.join(dirpath, 'config.yaml'))

    # INCOMPLETE BELOW
    ##############################################################################
    # if config.get('feature_selection'):
    #     extract_features = FeatureExtraction()
    #     extract_features.feature_scoring()
    #     extract_features.top_feature_extraction()
    #     extract_features.write_top_features_subset_data()

    # final_model_trainer = ModelTrainer()
    # final_model_trainer.train()

    # analyser = DownstreamAnalysis()
    # analyser.generate_analysis()
