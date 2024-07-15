import json
from os import path
from typing import Union

import pandas as pd
from pandas import DataFrame

from ..utils import read_json, dump_json

def extract_top_k_features(feature_class_weights: DataFrame,
                           k: int = None,
                           aggregation_strategy: str = 'mean',
                           dirpath: str = '.',
                           save_features: bool = True) -> Union[list[str], dict]:
    """Extract top k features from weight matrix trained on chunked features

    Args:
        feature_class_weights: DataFrame object containing weights of all features across all classes
        k: number of features to select from all_features
        aggregation_strategy: stratergy to aggregate features from each class, default: 'mean' 
        dirpath: directory to store all model_weights and top_features

    Returns:
        List of top k features
    """
    if k is None:
        k = len(feature_class_weights.columns)
    n_cls = len(feature_class_weights)

    if aggregation_strategy == 'mean':
        top_features_list = feature_class_weights.abs().mean().sort_values(
            ascending=False).reset_index()['index'][:k]
    elif aggregation_strategy == 'no_reduction':
        top_features_list = pd.DataFrame(
            columns=feature_class_weights.index.tolist())
        for i, category in enumerate(feature_class_weights.index.tolist()):
            top_features_list[category] = abs(
                feature_class_weights.iloc[i]).sort_values(
                    ascending=False).reset_index()['index'][:k]
    elif aggregation_strategy == 'classwise':
        top_features_list = dict()
        for i in range(n_cls):
            top_features_list[feature_class_weights.index[i]] = feature_class_weights.iloc[i,:].sort_values(
                        ascending=False).reset_index()['index'][:k].tolist()
    elif aggregation_strategy == 'mix':
        classwise_features = read_json(path.join(dirpath,'classwise_features.json'))
        classwise_features_list = []
        for key in classwise_features.keys():
            classwise_features_list += classwise_features[key]
        top_features_list = list(set(classwise_features_list))
        k_rem = k - len(top_features_list)
        print(top_features_list)
        feature_class_weights = feature_class_weights.drop(classwise_features_list, axis='columns')
        top_features_list += list(feature_class_weights.abs().mean().sort_values(
            ascending=False).reset_index()['index'][:k_rem])
    elif aggregation_strategy == 'difference':
        pass
    else:
        raise NotImplementedError(
            'Other aggregation strategies are not implemented yet...')

    if save_features:
        if aggregation_strategy != 'classwise':
            with open(path.join(dirpath,'top_features.txt'), 'w') as fh:
                fh.write('\n'.join(top_features_list) + '\n')
        else:
            dump_json(top_features_list, path.join(dirpath,'classwise_features.json'))

    return top_features_list