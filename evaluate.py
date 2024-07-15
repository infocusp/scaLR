import argparse
from copy import deepcopy
import os
from os import path
import sys

import torch
from torch import nn
import numpy as np

from config.utils import load_config
from config import default_config
from scalr import Trainer
from scalr.dataloader import simple_dataloader
from scalr.evaluation import get_predictions, accuracy, generate_and_save_classification_report, plot_roc_auc_curve, perform_differential_expression_analysis, generate_gene_recall_curve, save_top_genes_and_heatmap
from scalr.model import LinearModel
from scalr.utils import read_data, read_yaml, read_json, dump_yaml


def evaluate(config, log=True):

    print('GPU: ', torch.cuda.is_available())

    device = config['device']
    dirpath = config['dirpath']
    exp_name = config['exp_name']
    exp_run = config['exp_run']

    evaluation_configs = config['evaluation']

    dirpath = path.join(dirpath, f'{exp_name}_{exp_run}')
    resultpath = path.join(dirpath, 'results')
    os.makedirs(resultpath, exist_ok=True)
    if 'metrics' in evaluation_configs:
        data_config = config['data']
        target = data_config['target']
        test_datapath = data_config['test_datapath']
        train_datapath = data_config.get('train_datapath')
        batch_size = evaluation_configs['batch_size']

        model_checkpoint = evaluation_configs['model_checkpoint']
        model_ = read_yaml(path.join(model_checkpoint, 'config.yml'))
        config['model'] = model_
        model_type = model_['type']
        model_hp = model_['hyperparameters']

        # TODO: add absl.logging functionality
        if log:
            sys.stdout = open(f'{dirpath}/results/test.log', 'w')

        test_data = read_data(test_datapath)

        label_mappings = read_json(
            path.join(model_checkpoint, 'label_mappings.json'))

        if model_type == 'linear':
            model = LinearModel(**model_hp).to(device)
            model.load_state_dict(
                torch.load(path.join(model_checkpoint,
                                     'model.pt'))['model_state_dict'])

            test_dl = simple_dataloader(test_data, target, batch_size,
                                        label_mappings)

        # Evaluation
        id2label = label_mappings[target]['id2label']
        metrics = evaluation_configs['metrics']

        if metrics != ['deg']:
            test_labels, pred_labels, pred_probabilities = get_predictions(
                model, test_dl, device)

        if 'accuracy' in metrics:
            print('Accuracy: ', accuracy(test_labels, pred_labels))

        if 'report' in metrics:
            print('\nClassification Report:')
            generate_and_save_classification_report(test_labels,
                                                    pred_labels,
                                                    resultpath,
                                                    mapping=id2label)

        if 'roc_auc' in metrics:
            print("\nROC & AUC:")
            plot_roc_auc_curve(test_labels,
                               pred_probabilities,
                               resultpath,
                               mapping=id2label)
        if 'shap' in metrics:
            print("\nSHAP analysis:")
            shap_config = evaluation_configs.get('shap_config')

            if not shap_config:
                shap_config = deepcopy(default_config.shap_config)

            top_n = shap_config.get('top_n', 20)
            shap_batch_size = shap_config.get('batch_size', 1000)
            n_background_tensor = shap_config.get(
                'background_tensor', 200)
            early_stop_config = shap_config.get(
                'early_stop', default_config.shap_config['early_stop'])

            if train_datapath:
                train_data = read_data(train_datapath)
                train_dl = simple_dataloader(train_data, target, train_data.shape[0],
                                             label_mappings, shuffle=True)
            else:
                raise ValueError("Train data path required for SHAP analysis.")

            shap_test_dl = simple_dataloader(test_data, target, shap_batch_size,
                                        label_mappings, shuffle=True)

            save_top_genes_and_heatmap(model, train_dl, shap_test_dl, id2label,
                                       resultpath, early_stop_config, device,
                                       top_n, n_background_tensor)

    if 'deg_config' in evaluation_configs:
        assert config['data'], "Input data unavailable for deg analysis"
        assert 'full_datapath' in config['data'], "Required full_datapath for deg analysis"
        full_datapath = config['data'].get('full_datapath')
        ad_for_deg = read_data(full_datapath)
        perform_differential_expression_analysis(
            ad_for_deg,
            **evaluation_configs['deg_config'],
            dirpath=resultpath)

    if 'gene_recall' in evaluation_configs and evaluation_configs[
            'gene_recall']:
        print('Starting gene recall curve analysis.')
        generate_gene_recall_curve(evaluation_configs['gene_recall'],
                                   resultpath=resultpath)
        print('Gene recall curves generated.')

    dump_yaml(config, path.join(dirpath, 'config.yml'))
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
