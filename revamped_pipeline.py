import argparse
import os
import random
import sys

import numpy as np
import torch

from _scalr.utils.config import load_config
from data_ingestion import Dataingestion
from feature_extraction import FeatureExtraction
from model_trainer import modelTrainer

# from evaluate import evaluate

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

    dirpath = config['dirpath']
    exp_name = config['exp_name']
    exp_run = config['exp_run']

    dirpath = os.path.join(dirpath, f'{exp_name}_{exp_run}')
    
    #PIPELINE RUN!

    ingest_data = Dataingestion()
    ingest_data.tvt_split()
    ingest_data.preprocess()

    extract_features = FeatureExtraction()
    extract_features.feature_scoring()
    extract_features.top_feature_extraction()
    extract_features.write_top_features_subset_data()
    
    final_model_trainer = modelTrainer()
    final_model_trainer.train()
    
    

if __name__ == '__main__':
    main()
