import argparse
import os
import random
import sys

import numpy as np
import torch

from _scalr.utils.config import load_config
from _scalr.data_ingestion import DataIngestion
from _scalr.feature_extraction import FeatureExtraction
from _scalr.model_trainer import ModelTrainer
from _scalr.downstream_analysis import DownstreamAnalysis

def set_seed(seed: int):
    """To set seed for reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    os.makedirs(dirpath, exist_ok=True)
    
    
    #PIPELINE RUN
    
    ingest_data = DataIngestion()
    ingest_data.tvt_split()
    ingest_data.preprocess()

    extract_features = FeatureExtraction()
    extract_features.feature_scoring()
    extract_features.top_feature_extraction()
    extract_features.write_top_features_subset_data()
    
    final_model_trainer = ModelTrainer()
    final_model_trainer.train()
    
    analyser = DownstreamAnalysis()
    analyser.generate_analysis()
