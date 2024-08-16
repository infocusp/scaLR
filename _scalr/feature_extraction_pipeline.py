from copy import deepcopy
import os
from os import path

import numpy as np
import pandas as pd

from _scalr.feature import FeatureChunking
from _scalr.feature.scoring import build_scorer
from _scalr.feature.selector import build_selector
from _scalr.utils import load_train_val_data_from_config
from _scalr.utils import read_data
from _scalr.utils import write_chunkwise_data
from _scalr.utils import write_data


class FeatureExtractionPipeline:

    def __init__(self, feature_selction_config, data_config, dirpath, device):
        '''
        Feature extraction is done in 4 steps:
        1. Model(s) training on chunked/all features
        2. Class X Feature scoring
        3. Top features extraction
        4. Feature subset data writing
        '''
        self.feature_selection_config = deepcopy(feature_selction_config)
        self.data_config = deepcopy(data_config)
        self.dirpath = path.join(dirpath, 'feature_extraction')
        os.makedirs(self.dirpath, exist_ok=True)
        self.device = device

    def feature_chunked_model_training(self):

        self.train_data, self.val_data = load_train_val_data_from_config(
            self.data_config)
        feature_chunksize = self.feature_selection_config.get(
            'feature_chunksize', len(self.val_data.var_names))

        chunk_model_config = self.feature_selection_config.get('model')
        chunk_model_train_config = self.feature_selection_config.get(
            'model_train_config')
        target = self.data_config.get('target')
        mappings = read_data(self.data_config['label_mappings'])

        chunked_features_model_trainer = FeatureChunking(
            feature_chunksize, chunk_model_config, chunk_model_train_config,
            self.train_data, self.val_data, target, mappings, self.dirpath,
            self.device)

        self.chunked_models = chunked_features_model_trainer.train_chunked_models(
        )
        chunk_model_config, chunk_model_train_config = chunked_features_model_trainer.get_updated_configs(
        )
        self.feature_selection_config['model'] = chunk_model_config
        self.feature_selection_config[
            'model_train_config'] = chunk_model_train_config

        return self.chunked_models

    def feature_scoring(self):
        """To generate scores of each feature for each class using a scorer
        and chunked models
        """
        scorer, scorer_config = build_scorer(
            deepcopy(
                self.feature_selection_config.get('scoring_config',
                                                  dict(name='LinearScorer'))))
        self.feature_selection_config['scoring_config'] = scorer_config

        all_scores = []
        for model in self.chunked_models:
            score = scorer.generate_scores(model, self.train_data,
                                           self.val_data)
            all_scores.append(score)

        all_scores = np.concatenate(all_scores, axis=1)
        self.train_data.var_names.name = "index"
        class_labels = read_data(self.data_config['label_mappings'])[
            self.data_config.get('target')]['id2label']

        self.score_matrix = pd.DataFrame(all_scores,
                                         columns=self.train_data.var_names,
                                         index=class_labels)
        self.score_matrix.to_csv(path.join(self.dirpath, 'score_matrix.csv'))
        return self.score_matrix

    def top_feature_extraction(self):
        selector_config = self.feature_selection_config.get(
            'features_selector', dict(name='AbsMean'))
        selector, selector_config = build_selector(selector_config)
        self.feature_selection_config['features_selector'] = selector_config

        self.top_features = selector.get_feature_list(self.score_matrix)
        write_data(self.top_features,
                   path.join(self.dirpath, 'top_features.json'))

        return self.top_features

    def write_top_features_subset_data(self):
        datapath = self.data_config['train_val_test'].get('final_datapaths')
        feature_subset_datapath = path.join(self.dirpath, 'feature_subset_data')
        os.makedirs(feature_subset_datapath, exist_ok=True)

        for split in ['train', 'val', 'test']:

            split_datapath = path.join(datapath, split)
            split_feature_subset_datapath = path.join(feature_subset_datapath,
                                                      split)
            sample_chunksize = self.data_config.get('sample_chunksize')
            write_chunkwise_data(split_datapath,
                                 sample_chunksize,
                                 split_feature_subset_datapath,
                                 feature_inds=self.top_features)

        self.data_config['train_val_test'][
            'feature_subset_datapaths'] = feature_subset_datapath

    def get_updated_config(self):
        return self.feature_selection_config, self.data_config
