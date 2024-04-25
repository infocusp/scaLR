import os
import sys
import argparse
import torch
print('GPU: ' ,torch.cuda.is_available())
from torch import nn
import numpy as np
from scp.utils import load_config, read_data, read_yaml, read_json
from scp.tokenizer import GeneVocab
from scp.data import simpleDataLoader, transformerDataLoader
from scp.model import LinearModel, TransformerModel
from scp.evaluation import predictions, accuracy, report
from scp import Trainer

def main():
    # Parser to take in config file path and logging [enabled, disabled]
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', type=str, help='config.yml file')
    parser.add_argument('-l','--log', action='store_true', help='Store train-logs')

    args = parser.parse_args()
    
    # load config
    config = load_config(args.config)
    device = config['device']
    filepath = config['filepath']
    exp_name = config['exp_name']
    exp_run = config['exp_run']

    # create results directory
    filepath = f'{filepath}/{exp_name}_{exp_run}'
    os.makedirs(f'{filepath}/results', exist_ok=True)

    # Data
    data_config = config['data']
    target = data_config['target']
    test_datapath = data_config['test_datapath']

    evaluation_configs = config['evaluation']
    batch_size = evaluation_configs['batch_size']

    # Model params
    model_checkpoint = evaluation_configs['model_checkpoint']
    if model_checkpoint is None:
        model_checkpoint = f'{filepath}/best_model' 
    model_ = read_yaml(f'{model_checkpoint}/config.yml')
    model_type = model_['type']
    model_hp = model_['hyperparameters']

    # logging
    if args.log:
        sys.stdout = open(f'{filepath}/results/test.log','w')
    
    # loading data
    test_data = read_data(test_datapath)

    if data_config['use_top_features'] is True:
        with open(f'{filepath}/feature_selection/top_features.txt', 'r') as fh:
            top_features = list(map(lambda x:x[:-1], fh.readlines()))
    elif data_config['use_top_features'] is not None:
        with open(data_config['use_top_features'], 'r') as fh:
            top_features = list(map(lambda x:x[:-1], fh.readlines()))
    
    if data_config['use_top_features'] is not None:
        top_features_indices = sorted([test_data.var_names.tolist().index(feature) for feature in top_features])
        test_data = test_data[:,top_features_indices]
    
    label_mappings = read_json(f'{filepath}/label_mappings.json')

    # Linear model creation (and loading checkpoint model weights) and dataloaders
    if model_type == 'linear':
        features = model_hp['layers']
        dropout = model_hp['dropout']
        model = LinearModel(features, dropout).to(device)
        model.load_state_dict(torch.load(f'{model_checkpoint}/model.pt')['model_state_dict'])
        
        test_dl = simpleDataLoader(test_data, target, batch_size, label_mappings)

    # Transformer model creation (and loading checkpoint model weights) and dataloaders
    elif model_type == 'transformer':    
        # create vocab
        pad_token = "<pad>"
        cls_token = "<cls>"
        special_tokens = [pad_token, cls_token]
        pad_value = -2

        genes = test_data.var.index.tolist()
        vocab = GeneVocab.from_file(f'{model_checkpoint}/vocab.json')
        gene_ids = np.array(vocab(genes), dtype=int)
        ntokens=len(vocab)

        # model hyperparamters
        dim = model_hp['dim']
        nlayers = model_hp['nlayers']
        nheads = model_hp['nheads']
        dropout = model_hp['dropout']
        n_cls = model_hp['n_cls']
        decoder_layers = [dim, dim, n_cls]

        model = TransformerModel(
            ntokens,
            vocab,
            dim,
            nheads,
            nlayers,
            dropout,
            decoder_layers,
            pad_token
        ).to(device)

        model.load_state_dict(torch.load(f'{model_checkpoint}/model.pt')['model_state_dict'])

        #Tokenization
        prep = config['transformer_preprocessing']
        value_bin = prep['value_bin']
        n_bins = prep['n_bins']
        append_cls = prep['append_cls']
        include_zero_gene = prep['include_zero_gene']
        max_len = prep['max_len']

        test_dl = transformerDataLoader(
            test_data,
            target,
            batch_size,
            label_mappings,
            value_bin,
            n_bins,
            gene_ids,
            max_len,
            vocab,
            pad_token,
            pad_value,
            append_cls,
            include_zero_gene
        )

    # Evaluation
    id2label = label_mappings[target]['id2label']
    metrics = evaluation_configs['metrics']

    # Running inference on test data
    test_labels, pred_labels = predictions(model, test_dl, device)

    # Accuracy
    if 'accuracy' in metrics:
        print('Accuracy: ', accuracy(test_labels, pred_labels))
    
    # Classification Report
    if 'report' in metrics:
        print('\nClassification Report:')
        report(test_labels, pred_labels, f'{filepath}/results', mapping=id2label)

    #Heatmap
    # top_50_weights, top_50_genes = top_50_heatmap(model, f'{filepath}/results', classes=id2label ,genes=genes)


if __name__ == '__main__':
    main()




















