"""This file contains the implementation of feature subsetting, model training followed by top feature extraction."""

from copy import deepcopy
import os
from os import path
from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import numpy as np
import pandas as pd
from torch import nn

from scalr.feature import FeatureSubsetting
from scalr.feature.scoring import build_scorer
from scalr.feature.selector import build_selector
from scalr.utils import FlowLogger
from scalr.utils import load_train_val_data_from_config
from scalr.utils import read_data
from scalr.utils import write_chunkwise_data
from scalr.utils import write_data


class FeatureExtractionPipeline:

    def __init__(self, feature_selection_config, dirpath, device):
        """Initialize required parameters for feature selection.

        Feature extraction is done in 4 steps:
        1. Model(s) training on chunked/all features
        2. Class X Feature scoring
        3. Top features extraction
        4. Feature subset data writing

        Args:
            feature_selection_config: Feature selection config.
            dirpath: Path to load data from.
        """
        self.flow_logger = FlowLogger('FeatureExtraction')

        self.feature_selection_config = deepcopy(feature_selection_config)
        self.device = device

        self.dirpath = dirpath
        os.makedirs(dirpath, exist_ok=True)

    def load_data_and_targets_from_config(self, data_config: dict):
        """A function to load data and targets from data config.

        Args:
            data_config: Data config.
        """
        self.train_data, self.val_data = load_train_val_data_from_config(
            data_config)
        self.target = data_config.get('target')
        self.mappings = read_data(data_config['label_mappings'])
        self.sample_chunksize = data_config.get('sample_chunksize')

    def set_data_and_targets(self,
                             train_data: Union[AnnData, AnnCollection],
                             val_data: Union[AnnData, AnnCollection],
                             target: Union[str, list[str]],
                             mappings: dict,
                             sample_chunksize: int = None):
        """A function to set data when you don't use data directly from config,
        but rather by other sources like feature subsetting, etc.

        Args:
            train_data (Union[AnnData, AnnCollection]): Training data.
            val_data (Union[AnnData, AnnCollection]): Validation data.
            target (Union[str, list[str]]): Target columns name(s).
            mappings (dict): Mapping of a column value to ids
                            eg. mappings[column_name][label2id] = {A: 1, B:2, ...}.
            sample_chunksize (int): Chunks of samples to be loaded in memory at once.
        """
        self.train_data = train_data
        self.val_data = val_data
        self.target = target
        self.mappings = mappings

    def feature_subsetted_model_training(self) -> list[nn.Module]:
        """This function train models on subsetted data containing `feature_subsetsize` genes."""

        self.flow_logger.info('Feature subset models training')

        self.feature_subsetsize = self.feature_selection_config.get(
            'feature_subsetsize', len(self.val_data.var_names))
        self.num_workers = self.feature_selection_config.get('num_workers', 1)

        chunk_model_config = self.feature_selection_config.get('model')
        chunk_model_train_config = self.feature_selection_config.get(
            'model_train_config')

        chunked_features_model_trainer = FeatureSubsetting(
            self.feature_subsetsize, chunk_model_config,
            chunk_model_train_config, self.train_data, self.val_data,
            self.target, self.mappings, self.dirpath, self.device,
            self.num_workers, self.sample_chunksize)

        if self.num_workers > 1:
            chunked_features_model_trainer.write_feature_subsetted_data()

        self.chunked_models = chunked_features_model_trainer.train_chunked_models(
        )
        chunk_model_config, chunk_model_train_config = chunked_features_model_trainer.get_updated_configs(
        )
        self.feature_selection_config['model'] = chunk_model_config
        self.feature_selection_config[
            'model_train_config'] = chunk_model_train_config

        return self.chunked_models

    def set_model(self, models: list[nn.Module]):
        """A function to set the trained model for downstream feature tasks."""
        self.chunked_models = models

    def feature_scoring(self) -> pd.DataFrame:
        """A function to generate scores of each feature for each class using a scorer
        and chunked models.
        """
        self.flow_logger.info('Feature scoring')

        scorer, scorer_config = build_scorer(
            deepcopy(self.feature_selection_config.get('scoring_config')))
        self.feature_selection_config['scoring_config'] = scorer_config

        all_scores = []
        if not getattr(self, 'feature_subsetsize', None):
            self.feature_subsetsize = self.train_data.shape[1]

        # TODO: Parallelize feature scoring
        for i, (model) in enumerate(self.chunked_models):
            subset_train_data = self.train_data[:, i *
                                                self.feature_subsetsize:(i +
                                                                         1) *
                                                self.feature_subsetsize]
            subset_val_data = self.val_data[:, i *
                                            self.feature_subsetsize:(i + 1) *
                                            self.feature_subsetsize]
            score = scorer.generate_scores(model, subset_train_data,
                                           subset_val_data, self.target,
                                           self.mappings)

            all_scores.append(score[:self.feature_subsetsize])

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
        """A function to set score_matrix for feature extraction."""
        self.score_matrix = score_matrix

    def top_feature_extraction(self) -> Union[list[str], dict]:
        """A function to get top features using `Selector`."""

        self.flow_logger.info('Top features extraction')

        selector_config = self.feature_selection_config.get(
            'features_selector', dict(name='AbsMean'))
        selector, selector_config = build_selector(selector_config)
        self.feature_selection_config['features_selector'] = selector_config

        self.top_features = selector.get_feature_list(self.score_matrix)
        write_data(self.top_features,
                   path.join(self.dirpath, 'top_features.json'))

        return self.top_features

    def write_top_features_subset_data(self, data_config: dict) -> dict:
        """A function to write top features subset data onto disk
        and return updated data_config.

        Args:
            data_config: Data config.
        """

        self.flow_logger.info('Writing feature-subset data onto disk')

        datapath = data_config['train_val_test'].get('final_datapaths')

        feature_subset_datapath = path.join(self.dirpath, 'feature_subset_data')
        os.makedirs(feature_subset_datapath, exist_ok=True)

        test_data = read_data(path.join(datapath, 'test'))
        splits = {
            'train': self.train_data,
            'val': self.val_data,
            'test': test_data
        }

        sample_chunksize = data_config.get('sample_chunksize')
        num_workers = data_config.get('num_workers')

        for split, split_data in splits.items():

            split_feature_subset_datapath = path.join(feature_subset_datapath,
                                                      split)
            write_chunkwise_data(split_data,
                                 sample_chunksize,
                                 split_feature_subset_datapath,
                                 feature_inds=self.top_features,
                                 num_workers=num_workers)

        data_config['train_val_test'][
            'feature_subset_datapaths'] = feature_subset_datapath

        return data_config

    def get_updated_config(self) -> dict:
        """This function returns updated configs."""
        return self.feature_selection_config
