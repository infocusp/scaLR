import argparse
import os
import random
import sys

import numpy as np
import torch

from _scalr.utils.config import load_config
from _scalr.utils import set_seed
from projects.biocusp.scripts._scalr.data_ingestion_pipeline import DataIngestionPipeline
from projects.biocusp.scripts._scalr.feature_extraction_pipeline import FeatureExtraction
from projects.biocusp.scripts._scalr.model_training_pipeline import ModelTrainingPipeline
from projects.biocusp.scripts._scalr.downstream_analysis_pipeline import DownstreamAnalysis


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

    config = load_config(args.config)
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
    ingest_data.generate_label_mappings()
    config = ingest_data(config)

    if config.get('final_training'):
        model_trainer = ModelTrainingPipeline(
            dirpath, config['final_training']['model'],
            config['final_training']['training_config'], config['data'],
            device)

        model_trainer.load_data_from_config()
        model_trainer.train()

    # INCOMPLETE BELOW
    ##############################################################################
    if config.get('feature_selection'):
        extract_features = FeatureExtraction()
        extract_features.feature_scoring()
        extract_features.top_feature_extraction()
        extract_features.write_top_features_subset_data()

    final_model_trainer = ModelTrainer()
    final_model_trainer.train()

    analyser = DownstreamAnalysis()
    analyser.generate_analysis()
