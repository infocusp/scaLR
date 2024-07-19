import json
from os import path
from typing import Union

import pandas as pd
from pandas import DataFrame

from ..utils import read_json, dump_json


def extract_top_k_features(
        feature_class_weights: DataFrame,
        k: int = None,
        aggregation_strategy: str = 'mean',
        dirpath: str = '.',
        save_features: bool = True) -> Union[list[str], dict]:
    """Extract top k features from weight matrix trained on chunked features
    Scoring of features is done using the aggregation stratergy

    Args:
        feature_class_weights: DataFrame object containing weights of all features across all classes
        k: number of features to select from all_features
        aggregation_strategy: stratergy to aggregate features from each class, default: 'mean' 
        dirpath: directory to store all model_weights and top_features
        save_features: flag to store the features selected at `dirpath`

    Returns:
        List of top k features
    """
    if k is None:
        k = len(feature_class_weights.columns)
    n_cls = len(feature_class_weights)

    # Take the mean of absolute values of weights across all classes to get score for each feature
    if aggregation_strategy == 'mean':
        top_features_list = feature_class_weights.abs().mean().sort_values(
            ascending=False).reset_index()['index'][:k]

    # TODO: Redundant with classwise. Remove and replace in gene-recal
    # Take the top absolute weighted features for each class. This is a list on impactful features for the particular class.
    # Consists of promoters and inhibitors
    elif aggregation_strategy == 'no_reduction':
        top_features_list = pd.DataFrame(
            columns=feature_class_weights.index.tolist())
        for i, category in enumerate(feature_class_weights.index.tolist()):
            top_features_list[category] = abs(
                feature_class_weights.iloc[i]).sort_values(
                    ascending=False).reset_index()['index'][:k]

    # Take the top weighted features for each class. This is a list on promoting features for the particular class
    elif aggregation_strategy == 'classwise_promoters':
        top_features_list = dict()
        for i in range(n_cls):
            top_features_list[feature_class_weights.index[
                i]] = feature_class_weights.iloc[i, :].sort_values(
                    ascending=False).reset_index()['index'][:k].tolist()

    # Take the top absolute weighted features for each class. This is a list on impactful features for the particular class.
    # Consists of promoters and inhibitors
    elif aggregation_strategy == 'classwise':
        top_features_list = dict()
        for i in range(n_cls):
            top_features_list[feature_class_weights.index[i]] = abs(
                feature_class_weights).iloc[i, :].sort_values(
                    ascending=False).reset_index()['index'][:k].tolist()

    # Take the top-100 `classwise` features + `mean` features.
    elif aggregation_strategy == 'mix':
        classwise_features = extract_top_k_features(
            feature_class_weights,
            k=100,
            aggregation_strategy='classwise',
            save_features=False)
        classwise_features_list = []
        for key in classwise_features.keys():
            classwise_features_list += classwise_features[key]
        top_features_list = list(set(classwise_features_list))
        k_rem = k - len(top_features_list)
        feature_class_weights = feature_class_weights.drop(
            classwise_features_list, axis='columns')
        top_features_list += list(
            feature_class_weights.abs().mean().sort_values(
                ascending=False).reset_index()['index'][:k_rem])

    # ONLY Applicable in binary classification
    # Take difference between weights of a feature to get the 'distance' or 'impact' created by a feature.
    elif aggregation_strategy == 'difference':
        # TODO: implement difference based selection for binary classification
        pass
    else:
        raise NotImplementedError(
            'Other aggregation strategies are not implemented yet...')

    if save_features:
        if 'classwise' not in aggregation_strategy:
            with open(path.join(dirpath, 'top_features.txt'), 'w') as fh:
                fh.write('\n'.join(top_features_list) + '\n')
        else:
            dump_json(top_features_list, path.join(dirpath, 'biomarkers.json'))

    return top_features_list
