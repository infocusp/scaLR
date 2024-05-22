import os
import sys
import argparse

import torch
from torch import nn
import numpy as np

from scalr.utils import load_config, read_data, read_yaml, dump_yaml, dump_json
from scalr.dataloader import simple_dataloader
from scalr.model import LinearModel
from scalr import Trainer


def train(config, log=True):

    print('GPU: ', torch.cuda.is_available())

    device = config['device']
    dirpath = config['dirpath']
    exp_name = config['exp_name']
    exp_run = config['exp_run']

    dirpath = f'{dirpath}/{exp_name}_{exp_run}'
    os.makedirs(f'{dirpath}/', exist_ok=True)

    data_config = config['data']
    target = data_config['target']
    train_datapath = data_config['train_datapath']
    val_datapath = data_config['val_datapath']

    train_config = config['training']
    optimizer = train_config['opt']
    lossfunc = train_config['loss']
    batch_size = train_config['batch_size']
    lr = train_config['lr']
    l2 = float(train_config['l2'])
    epochs = train_config['epochs']
    callbacks = train_config['callbacks']

    resume_from_checkpoint = config['model']['resume_from_checkpoint']
    if resume_from_checkpoint:
        model_checkpoint = config['model']['start_checkpoint']
        model_ = read_yaml(f'{model_checkpoint}/config.yml')
        config['model']['type'] = model_['type']
        config['model']['hyperparameters'] = model_['hyperparameters']
    else:
        model_checkpoint = None
        config['model']['start_checkpoint'] = None
    model_type = config['model']['type']
    model_hp = config['model']['hyperparameters']

    if 'evaluation' in config:
        config['evaluation']['model_checkpoint'] = f'{dirpath}/best_model'
    else:
        config['evaluation'] = {'model_checkpoint': f'{dirpath}/best_model'}

    # TODO: add absl.logging functionality
    if log:
        sys.stdout = open(f'{dirpath}/train.log', 'w')

    train_data = read_data(train_datapath)
    val_data = read_data(val_datapath)

    # Create mappings for targets to be used by testing
    label_mappings = {}
    label_mappings[target] = {}
    id2label = train_data.obs[target].astype(
        'category').cat.categories.tolist()
    label2id = {id2label[i]: i for i in range(len(id2label))}
    label_mappings[target]['id2label'] = id2label
    label_mappings[target]['label2id'] = label2id

    # model creation and dataloaders
    if model_type == 'linear':
        # features = model_hp['layers']
        # dropout = model_hp['dropout']
        model = LinearModel(**model_hp)

        train_dl = simple_dataloader(train_data, target, batch_size,
                                     label_mappings)
        val_dl = simple_dataloader(val_data, target, batch_size,
                                   label_mappings)
    else:
        raise NotImplementedError('Only `linear` available as options!')

    # Training

    if optimizer == 'adam':
        opt = torch.optim.Adam
    elif optimizer == 'sgd':
        opt = torch.optim.SGD
    else:
        raise NotImplementedError(
            'Only `adam` and `sgd` available as options!')

    if lossfunc == 'log':
        loss_fn = nn.CrossEntropyLoss()
    elif lossfunc == 'weighted_log':
        weights = 1 / torch.as_tensor(
            train_data.obs[target].value_counts()[id2label],
            dtype=torch.float32)
        total_sum = weights.sum()
        weights /= total_sum
        loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        raise NotImplementedError(
            'Only `log` and `weighted_log` available as options!')

    trainer = Trainer(model, opt, lr, l2, loss_fn, callbacks, device, dirpath,
                      model_checkpoint)
    trainer.train(epochs, train_dl, val_dl)

    dump_yaml(config['model'], f'{dirpath}/best_model/config.yml')
    dump_json(label_mappings, f'{dirpath}/best_model/label_mappings.json')
    dump_yaml(config, f'{dirpath}/config.yml')
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='config.yml file')
    parser.add_argument('-l',
                        '--log',
                        action='store_true',
                        help='Store train-logs')

    args = parser.parse_args()

    # load config file
    config = load_config(args.config)

    train(config, args.log)
