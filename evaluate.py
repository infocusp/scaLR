import os
import sys
import argparse

import torch
from torch import nn
import numpy as np

from scalr.utils import load_config, read_data, read_yaml, read_json, dump_yaml
from scalr.dataloader import simple_dataloader
from scalr.model import LinearModel
from scalr.evaluation import get_predictions, accuracy, generate_and_save_classification_report, roc_auc
from scalr import Trainer


def evaluate(config, log=True):

    print('GPU: ', torch.cuda.is_available())

    device = config['device']
    dirpath = config['dirpath']
    exp_name = config['exp_name']
    exp_run = config['exp_run']

    data_config = config['data']
    target = data_config['target']
    test_datapath = data_config['test_datapath']

    evaluation_configs = config['evaluation']
    batch_size = evaluation_configs['batch_size']
    if 'metrics' not in evaluation_configs:
        return config

    dirpath = f'{dirpath}/{exp_name}_{exp_run}'
    os.makedirs(f'{dirpath}/results', exist_ok=True)

    model_checkpoint = evaluation_configs['model_checkpoint']
    model_ = read_yaml(f'{model_checkpoint}/config.yml')
    config['model'] = model_
    model_type = model_['type']
    model_hp = model_['hyperparameters']

    # TODO: add absl.logging functionality
    if log:
        sys.stdout = open(f'{dirpath}/results/test.log', 'w')

    test_data = read_data(test_datapath)

    label_mappings = read_json(f'{model_checkpoint}/label_mappings.json')

    if model_type == 'linear':
        model = LinearModel(**model_hp).to(device)
        model.load_state_dict(
            torch.load(f'{model_checkpoint}/model.pt')['model_state_dict'])

        test_dl = simple_dataloader(test_data, target, batch_size,
                                    label_mappings)

    # Evaluation
    id2label = label_mappings[target]['id2label']
    metrics = evaluation_configs['metrics']

    test_labels, pred_labels, pred_probabilities = get_predictions(
        model, test_dl, device)

    if 'accuracy' in metrics:
        print('Accuracy: ', accuracy(test_labels, pred_labels))

    if 'report' in metrics:
        print('\nClassification Report:')
        generate_and_save_classification_report(test_labels,
                                                pred_labels,
                                                f'{dirpath}/results',
                                                mapping=id2label)

    if 'roc_auc' in metrics:
        print("\nROC & AUC:")
        roc_auc(test_labels,
                pred_probabilities,
                f'{dirpath}/results',
                mapping=id2label)

    dump_yaml(config, f'{dirpath}/config.yml')
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='config.yml file')
    parser.add_argument('-l',
                        '--log',
                        action='store_true',
                        help='Store evaluation-logs')

    args = parser.parse_args()

    config = load_config(args.config)

    evaluate(config, args.log)
