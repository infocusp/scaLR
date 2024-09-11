"""This file contains implementation for model training on feature subsets."""

from copy import deepcopy
import os
from os import path
from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
from torch import nn

from scalr.model_training_pipeline import ModelTrainingPipeline
from scalr.utils import EventLogger
from scalr.utils import FlowLogger


class FeatureSubsetting:
    """Class for FeatureSubsetting.

    It trains a model for each subsetted datasets, each
    containing `feature_subsetsize` genes as features.
    """

    def __init__(self,
                 feature_subsetsize: int,
                 chunk_model_config: dict,
                 chunk_model_train_config: dict,
                 train_data: Union[AnnData, AnnCollection],
                 val_data: Union[AnnData, AnnCollection],
                 target: str,
                 mappings: dict,
                 dirpath: str = None,
                 device: str = 'cpu'):
        """Initialize required parameters for feature subset training.

        Args:
            feature_subsetsize (int): Number of features in one subset.
            chunk_model_config (dict): Chunked model config.
            chunk_model_train_config (dict): Chunked model training config.
            train_data (Union[AnnData, AnnCollection]): Train dataset.
            val_data (Union[AnnData, AnnCollection]): Validation dataset.
            target (str): Target to train model.
            mappings (dict): mapping of target to labels.
            dirpath (str, optional): Dirpath to store chunked model weights. Defaults to None.
            device (str, optional): Device to train models on. Defaults to 'cpu'.
        """
        self.event_logger = EventLogger('FeatureSubsetting')

        self.feature_subsetsize = feature_subsetsize
        self.chunk_model_config = chunk_model_config
        self.chunk_model_train_config = chunk_model_train_config
        self.train_data = train_data
        self.val_data = val_data
        self.target = target
        self.mappings = mappings
        self.dirpath = dirpath
        self.device = device

    def train_chunked_models(self) -> list[nn.Module]:
        """Trains a model for each subset data.

        Returns:
            list[nn.Module]: List of models for each subset.
        """
        self.event_logger.info('Feature subset models training')
        models = []
        chunked_models_dirpath = path.join(self.dirpath, 'chunked_models')
        os.makedirs(chunked_models_dirpath, exist_ok=True)

        i = 0
        for start in range(0, len(self.train_data.var_names),
                           self.feature_subsetsize):
            self.event_logger.info(f'\nChunk {i}')
            chunk_dirpath = path.join(chunked_models_dirpath, str(i))
            os.makedirs(chunk_dirpath, exist_ok=True)
            i += 1

            train_features_subset = self.train_data[:, start:start +
                                                    self.feature_subsetsize]
            val_features_subset = self.val_data[:, start:start +
                                                self.feature_subsetsize]

            chunk_model_config = deepcopy(self.chunk_model_config)

            model_trainer = ModelTrainingPipeline(chunk_model_config,
                                                  self.chunk_model_train_config,
                                                  chunk_dirpath, self.device)

            model_trainer.set_data_and_targets(train_features_subset,
                                               val_features_subset, self.target,
                                               self.mappings)
            model_trainer.build_model_training_artifacts()
            best_model = model_trainer.train()

            self.chunk_model_config, self.chunk_model_train_config = model_trainer.get_updated_config(
            )

            models.append(best_model)

        return models

    def get_updated_configs(self):
        """Returns updated configs."""
        return self.chunk_model_config, self.chunk_model_train_config
