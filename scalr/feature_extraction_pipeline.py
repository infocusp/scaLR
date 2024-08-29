from copy import deepcopy
import os
from os import path
from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import numpy as np
import pandas as pd
from torch import nn

from scalr.feature import FeatureChunking
from scalr.feature.scoring import build_scorer
from scalr.feature.selector import build_selector
from scalr.utils import EventLogger
from scalr.utils import FlowLogger
from scalr.utils import load_train_val_data_from_config
from scalr.utils import read_data
from scalr.utils import write_chunkwise_data
from scalr.utils import write_data


class FeatureExtractionPipeline:

    def __init__(self, feature_selection_config, dirpath, device):
        '''
        Feature extraction is done in 4 steps:
        1. Model(s) training on chunked/all features
        2. Class X Feature scoring
        3. Top features extraction
        4. Feature subset data writing
        '''
        self.flow_logger = FlowLogger('FeatureExtraction')

        self.feature_selection_config = deepcopy(feature_selection_config)
        self.device = device

        self.dirpath = dirpath
        os.makedirs(dirpath, exist_ok=True)

    def load_data_and_targets_from_config(self, data_config: dict):
        """load data and targets from data config"""
        self.train_data, self.val_data = load_train_val_data_from_config(
            data_config)
        self.target = data_config.get('target')
        self.mappings = read_data(data_config['label_mappings'])

    def set_data_and_targets(self, train_data: Union[AnnData, AnnCollection],
                             val_data: Union[AnnData, AnnCollection],
                             target: Union[str, list[str]], mappings: dict):
        """Useful when you don't use data directly from config, but rather by other
        sources like feature chunking, etc.

        Args:
            train_data (Union[AnnData, AnnCollection]): training data
            val_data (Union[AnnData, AnnCollection]): validation data
            target (Union[str, list[str]]): target columns name(s)
            mappings (dict): mapping of column value to ids
                            eg. mappings[column_name][label2id] = {A: 1, B:2, ...}
        """
        self.train_data = train_data
        self.val_data = val_data
        self.target = target
        self.mappings = mappings

    def feature_chunked_model_training(self):
        self.flow_logger.info('Feature chunked models training')

        feature_chunksize = self.feature_selection_config.get(
            'feature_chunksize', len(self.val_data.var_names))

        chunk_model_config = self.feature_selection_config.get('model')
        chunk_model_train_config = self.feature_selection_config.get(
            'model_train_config')

        chunked_features_model_trainer = FeatureChunking(
            feature_chunksize, chunk_model_config, chunk_model_train_config,
            self.train_data, self.val_data, self.target, self.mappings,
            self.dirpath, self.device)

        self.chunked_models = chunked_features_model_trainer.train_chunked_models(
        )
        chunk_model_config, chunk_model_train_config = chunked_features_model_trainer.get_updated_configs(
        )
        self.feature_selection_config['model'] = chunk_model_config
        self.feature_selection_config[
            'model_train_config'] = chunk_model_train_config

        return self.chunked_models

    def set_model(self, models: list[nn.Module]):
        self.chunked_models = models

    def feature_scoring(self):
        """To generate scores of each feature for each class using a scorer
        and chunked models
        """
        self.flow_logger.info('Feature scoring')

        scorer, scorer_config = build_scorer(
            deepcopy(self.feature_selection_config.get('scoring_config')))
        self.feature_selection_config['scoring_config'] = scorer_config

        all_scores = []
        feature_chunksize = self.feature_selection_config.get(
            'feature_chunksize', len(self.val_data.var_names))
        for model in self.chunked_models:
            score = scorer.generate_scores(model, self.train_data,
                                           self.val_data, self.target,
                                           self.mappings)
            all_scores.append(score[:feature_chunksize])

        columns = self.train_data.var_names
        columns.name = "index"
        class_labels = self.mappings[self.target]['id2label']
        all_scores = np.concatenate(all_scores, axis=1)
        all_scores = all_scores[:, :len(columns)]

        self.score_matrix = pd.DataFrame(all_scores,
                                         columns=columns,
                                         index=class_labels)
        write_data(self.score_matrix, path.join(self.dirpath,
                                                'score_matrix.csv'))
        return self.score_matrix

    def set_score_matrix(self, score_matrix: pd.DataFrame):
        self.score_matrix = score_matrix

    def top_feature_extraction(self):
        self.flow_logger.info('Top features extraction')

        selector_config = self.feature_selection_config.get(
            'features_selector', dict(name='AbsMean'))
        selector, selector_config = build_selector(selector_config)
        self.feature_selection_config['features_selector'] = selector_config

        self.top_features = selector.get_feature_list(self.score_matrix)
        write_data(self.top_features,
                   path.join(self.dirpath, 'top_features.json'))

        return self.top_features

    def write_top_features_subset_data(self, data_config):
        self.flow_logger.info('Writing feature-subset data onto disk')

        datapath = data_config['train_val_test'].get('final_datapaths')
        feature_subset_datapath = path.join(self.dirpath, 'feature_subset_data')
        os.makedirs(feature_subset_datapath, exist_ok=True)

        for split in ['train', 'val', 'test']:

            split_datapath = path.join(datapath, split)
            split_feature_subset_datapath = path.join(feature_subset_datapath,
                                                      split)
            sample_chunksize = data_config.get('sample_chunksize')
            write_chunkwise_data(split_datapath,
                                 sample_chunksize,
                                 split_feature_subset_datapath,
                                 feature_inds=self.top_features)

        data_config['train_val_test'][
            'feature_subset_datapaths'] = feature_subset_datapath

        return data_config

    def get_updated_config(self):
        return self.feature_selection_config
