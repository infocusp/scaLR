import os

import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
from torch import nn

from ..model import LinearModel
from ..trainer import Trainer
from ..dataloader import simple_dataloader


def feature_chunking(train_data,
                        val_data,
                        target: str,
                        model_config: dict,
                        chunksize: int,
                        dirpath: str,
                        device: str = 'cpu') -> list[str]:
    """ Feature selection using feature chunking approach.

        #TODO: add brief about approach
    
        Args:
            train_data: train_dataset (anndata oject)
            val_data: validation_dataset (anndata object)
            target: target class which is present in dataset.obs for classifcation training
            model_config: dict containing type of model, and it related config
            chunksize: number of features to take in one training instance
            dirpath: directory to store all model_weights and top_features
            device: [cpu/cuda] device to perform training on.

        Return:
            Weight matrix of all features across all classes
    """

    label_mappings = {}
    label_mappings[target] = {}
    id2label = train_data.obs[target].astype('category').cat.categories.tolist()
    label2id = {id2label[i]: i for i in range(len(id2label))}
    label_mappings[target]['id2label'] = id2label
    label_mappings[target]['label2id'] = label2id

    n_cls = len(id2label)
    epochs = model_config['params'].get('epochs', 25)
    batch_size = model_config['params'].get('batch_size', 15000)
    lr = model_config['params'].get('lr', 1e-2)
    l2 = model_config['params'].get('l2', 0.1)

    os.makedirs(f'{dirpath}/model_weights', exist_ok=True)

    best_model_weights = []
    i = 0
    for start in range(0, len(train_data.var_names), chunksize):

        train_features_subset = train_data[:, start:start + chunksize]
        val_features_subset = val_data[:, start:start + chunksize]

        train_dl = simple_dataloader(train_features_subset, target, batch_size,
                                     label_mappings)
        val_dl = simple_dataloader(val_features_subset, target, batch_size,
                                   label_mappings)

        model = LinearModel([len(train_features_subset.var_names), n_cls],
                            weights_init_zero=True)
        opt = torch.optim.SGD
        loss_fn = nn.CrossEntropyLoss()

        callbacks = {
            'model_checkpoint': {
                'interval': 0,
            },
            'early_stop': {
                'patience': 3,
                'min_delta': 1e-4
            }
        }

        dirpath_ = f'{dirpath}/model_weights/{i}'
        os.makedirs(f'{dirpath_}/best_model', exist_ok=True)

        trainer_ = Trainer(model, opt, lr, l2, loss_fn, callbacks, device,
                           dirpath_)
        trainer_.train(epochs, train_dl, val_dl)

        best_model_weights.append(f'{dirpath_}/best_model/model.pt')

        i += 1

    # Selecting top_k features
    feature_class_weights = pd.DataFrame()
    model_parent_path = f'{dirpath}/model_weights'

    all_weights = []
    # Loading models from each chunk and generating feature class weights matrix.
    for file in best_model_weights:

        weights = torch.load(
            file)['model_state_dict']['out_layer.weight'].cpu()
        all_weights.append(weights)

    full_weights_matrix = torch.cat(all_weights, dim=1)
    feature_class_weights = pd.DataFrame(full_weights_matrix,
                                         columns=train_data.var_names,
                                         index=id2label)

    # Storing feature class weights matrix.
    feature_class_weights.to_csv(f'{dirpath}/feature_class_weights.csv')

    return feature_class_weights

def extract_top_k_features(feature_class_weights: DataFrame,k: int,
                           aggregation_strategy:str='mean', dirpath:str='.'):
    """Extract top k features from weight matrix trained on chunked features

    Args:
        feature_class_weights: DataFrame object containing weights of all features across all classes
        k: number of features to select from all_features
        aggregation_strategy: stratergy to aggregate features from each class, default: 'mean' 
        dirpath: directory to store all model_weights and top_features

    Returns:
        List of top k features
    """
    
    if aggregation_strategy == 'mean':
        top_features_list = feature_class_weights.abs().mean().sort_values(
            ascending=False).reset_index()['index'][:k]
    else:
        raise NotImplementedError(
            'Other aggregation strategies are not implemented yet...')

    with open(f'{dirpath}/top_features.txt', 'w') as fh:
        fh.write('\n'.join(top_features_list) + '\n')

    return top_features_list
