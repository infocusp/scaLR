"""This file contains implementation for model training on feature subsets."""

from copy import deepcopy
import os
from os import path
from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
from joblib import delayed
from joblib import Parallel
from torch import nn

from scalr.model_training_pipeline import ModelTrainingPipeline
from scalr.utils import EventLogger
from scalr.utils import FlowLogger
from scalr.utils import read_data
from scalr.utils import write_chunkwise_data


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
                 device: str = 'cpu',
                 num_workers: int = 1,
                 sample_chunksize: int = None):
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
            num_workers (int, optional): Number of parallel processes to launch to train multiple
                                         feature subsets simultaneously. Defaults to using single
                                         process.
            sample_chunksize (int, optional): Chunks of samples to be loaded in memory at once. 
                                              Required when `num_workers` > 1.
        """
        self.feature_subsetsize = feature_subsetsize
        self.chunk_model_config = chunk_model_config
        self.chunk_model_train_config = chunk_model_train_config
        self.train_data = train_data
        self.val_data = val_data
        self.target = target
        self.mappings = mappings
        self.dirpath = dirpath
        self.device = device
        self.num_workers = num_workers if num_workers else 1
        self.sample_chunksize = sample_chunksize

        self.total_features = len(self.train_data.var_names)

        # Note that EventLogger does not work with parallel training
        # You may use tensorboard logging to track model training logs
        if self.num_workers == 1:
            self.event_logger = EventLogger('FeatureSubsetting')

    def write_feature_subsetted_data(self):
        """Write chunks of feature-subsetted data, to enable parallel training of models
        using different chunks of data."""
        if self.num_workers == 1:
            return

        self.feature_chunked_data_dirpath = path.join(self.dirpath,
                                                      'chunked_data')
        os.makedirs(self.feature_chunked_data_dirpath, exist_ok=True)

        i = 0
        for start in range(0, self.total_features, self.feature_subsetsize):

            feature_subset_inds = list(
                range(start,
                      min(start + self.feature_subsetsize,
                          self.total_features)))

            write_chunkwise_data(self.train_data,
                                 self.sample_chunksize,
                                 path.join(self.feature_chunked_data_dirpath,
                                           'train', str(i)),
                                 feature_inds=feature_subset_inds,
                                 num_workers=self.num_workers)

            write_chunkwise_data(self.val_data,
                                 self.sample_chunksize,
                                 path.join(self.feature_chunked_data_dirpath,
                                           'val', str(i)),
                                 feature_inds=feature_subset_inds,
                                 num_workers=self.num_workers)

            i += 1

        del self.train_data
        del self.val_data

    def train_chunked_models(self) -> list[nn.Module]:
        """Trains a model for each subset data.

        Returns:
            list[nn.Module]: List of models for each subset.
        """
        if self.num_workers == 1:
            self.event_logger.info('Feature subset models training')

        chunked_models_dirpath = path.join(self.dirpath, 'chunked_models')
        os.makedirs(chunked_models_dirpath, exist_ok=True)

        def train_chunked_model(i, start):
            if self.num_workers == 1:
                self.event_logger.info(f'\nChunk {i}')

            chunk_dirpath = path.join(chunked_models_dirpath, str(i))
            os.makedirs(chunk_dirpath, exist_ok=True)

            if self.num_workers > 1:
                train_features_subset = read_data(
                    path.join(self.feature_chunked_data_dirpath, 'train',
                              str(i)))
                val_features_subset = read_data(
                    path.join(self.feature_chunked_data_dirpath, 'val', str(i)))
            else:
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

            return i, best_model

        parallel = Parallel(n_jobs=self.num_workers)
        models = parallel(
            delayed(train_chunked_model)(i, start) for i, (start) in enumerate(
                range(0, self.total_features, self.feature_subsetsize)))

        # parallel loop returns all models with the chunk number, which is used to sort models in order
        # model[1] returns only the model, without the chunk number
        models = sorted(models)
        models = [model[1] for model in models]
        return models

    def get_updated_configs(self):
        """Returns updated configs."""
        return self.chunk_model_config, self.chunk_model_train_config
