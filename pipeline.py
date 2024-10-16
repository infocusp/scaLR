"""This file contains an implementation of end-to-end pipeline execution."""

import argparse
import logging
import os
from os import path
import random
import sys
from time import time

from memory_profiler import memory_usage
from memory_profiler import profile
import numpy as np
import torch

from scalr.data_ingestion_pipeline import DataIngestionPipeline
from scalr.eval_and_analysis_pipeline import EvalAndAnalysisPipeline
from scalr.feature_extraction_pipeline import FeatureExtractionPipeline
from scalr.model_training_pipeline import ModelTrainingPipeline
from scalr.utils import EventLogger
from scalr.utils import FlowLogger
from scalr.utils import read_data
from scalr.utils import set_seed
from scalr.utils import write_data


def get_args():
    """A function to get the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        help='config.yaml file path',
                        required=True)
    parser.add_argument('-l',
                        '--log',
                        action='store_true',
                        help='flag to store logs for the experiment')
    parser.add_argument('--level',
                        type=str,
                        default='INFO',
                        help='set the level of logging')
    parser.add_argument('--logpath',
                        type=str,
                        default=False,
                        help='path to store the logs')
    parser.add_argument('-m',
                        '--memoryprofiler',
                        action='store_true',
                        help='flag to get memory usage analysis')

    args = parser.parse_args()
    return args


# Uncomment `@profile` to get line-by-line memory analysis
# @profile
def pipeline(config, dirpath, device, flow_logger, event_logger):
    """A function that configures all components of the pipeline for end-to-end execution.
    
    Args:
        config: User config.
        dirpath: Path of root directory.
        flow_logger: Object for flow logger.
        event_logger: Object for event logger.
    """
    if config.get('data'):
        # Data ingestion.
        flow_logger.info('Data Ingestion pipeline running')
        event_logger.heading1('Data Ingestion')

        data_dirpath = path.join(dirpath, 'data')
        os.makedirs(data_dirpath, exist_ok=True)

        ingest_data = DataIngestionPipeline(config['data'], data_dirpath)
        ingest_data.generate_train_val_test_split()
        ingest_data.preprocess_data()
        if not config['data'].get('label_mappings'):
            ingest_data.generate_mappings()

        config['data'] = ingest_data.get_updated_config()
        write_data(config, path.join(dirpath, 'config.yaml'))
        del ingest_data

    if config.get('feature_selection'):
        # Feature selection.
        flow_logger.info('Feature Extraction pipeline running')
        event_logger.heading1('Feature Selection')

        feature_extraction_dirpath = path.join(dirpath, 'feature_extraction')
        os.makedirs(feature_extraction_dirpath, exist_ok=True)

        extract_features = FeatureExtractionPipeline(
            config['feature_selection'], feature_extraction_dirpath, device)
        extract_features.load_data_and_targets_from_config(config['data'])

        if not config['feature_selection'].get('score_matrix'):
            extract_features.feature_subsetted_model_training()
            extract_features.feature_scoring()
        else:
            extract_features.set_score_matrix(
                read_data(config['feature_selection'].get('score_matrix')))

        extract_features.top_feature_extraction()
        config['data'] = extract_features.write_top_features_subset_data(
            config['data'])

        config['feature_selection'] = extract_features.get_updated_config()
        write_data(config, path.join(dirpath, 'config.yaml'))
        del extract_features

    if config.get('final_training'):
        # Final model training.
        flow_logger.info('Final Model Training pipeline running')
        event_logger.heading1('Final Model Training')

        model_training_dirpath = path.join(dirpath, 'model')
        os.makedirs(model_training_dirpath, exist_ok=True)

        model_trainer = ModelTrainingPipeline(
            config['final_training']['model'],
            config['final_training']['model_train_config'],
            model_training_dirpath, device)

        model_trainer.load_data_and_targets_from_config(config['data'])
        model_trainer.build_model_training_artifacts()
        model_trainer.train()

        model_config, model_train_config = model_trainer.get_updated_config()
        config['final_training']['model'] = model_config
        config['final_training']['model_train_config'] = model_train_config
        write_data(config, path.join(dirpath, 'config.yaml'))
        del model_trainer

    if config.get('analysis'):
        # Analysis of trained model.
        flow_logger.info('Analysis pipeline running')
        event_logger.heading1('Analysis')

        analysis_dirpath = path.join(dirpath, 'analysis')
        os.makedirs(analysis_dirpath, exist_ok=True)

        if config.get('final_training'):
            config['analysis']['model_checkpoint'] = path.join(
                model_training_dirpath, 'best_model')

        analyser = EvalAndAnalysisPipeline(config['analysis'], analysis_dirpath,
                                           device)
        analyser.load_data_and_targets_from_config(config['data'])

        if config['analysis'].get('model_checkpoint'):
            analyser.evaluation_and_classification_report()

            if config['analysis'].get('gene_analysis'):
                analyser.gene_analysis()

        analyser.full_samples_downstream_anlaysis()
        analyser.test_samples_downstream_anlaysis()

        config['analysis'] = analyser.get_updated_config()
        write_data(config, path.join(dirpath, 'config.yaml'))
        del analyser

    return config


if __name__ == '__main__':
    set_seed(42)
    args = get_args()

    start_time = time()

    # Parsing config.
    config = read_data(args.config)

    # Setting experiment information form config.
    dirpath = config['experiment']['dirpath']
    exp_name = config['experiment']['exp_name']
    exp_run = config['experiment']['exp_run']
    dirpath = os.path.join(dirpath, f'{exp_name}_{exp_run}')
    device = config['device']

    # Logging.
    log = args.log
    if log:
        level = getattr(logging, args.level)
        logpath = args.logpath if args.logpath else path.join(
            dirpath, 'logs.txt')
    else:
        level = logging.CRITICAL
        logpath = None

    flow_logger = FlowLogger('ROOT', level)
    flow_logger.info(f'Experiment directory: `{dirpath}`')
    if os.path.exists(dirpath):
        flow_logger.warning('Experiment directory already exists!')

    os.makedirs(dirpath, exist_ok=True)

    event_logger = EventLogger('ROOT', level, logpath)

    kwargs = dict(config=config,
                  dirpath=dirpath,
                  device=device,
                  flow_logger=flow_logger,
                  event_logger=event_logger)

    if args.memoryprofiler:
        max_memory = memory_usage((pipeline, [], kwargs),
                                  max_usage=True,
                                  interval=0.5,
                                  max_iterations=1)
    else:
        pipeline(**kwargs)

    end_time = time()
    flow_logger.info(f'Total time taken: {end_time - start_time} s')

    event_logger.heading1('Runtime Analyis')
    event_logger.info(f'Total time taken: {end_time - start_time} s')

    if args.memoryprofiler:
        flow_logger.info(f'Maximum memory usage: {max_memory} MB')
        event_logger.info(f'Maximum memory usage: {max_memory} MB')
