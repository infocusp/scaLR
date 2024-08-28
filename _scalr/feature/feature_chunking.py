from copy import deepcopy
import os
from os import path
from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection

from _scalr.model_training_pipeline import ModelTrainingPipeline
from _scalr.utils import EventLogger
from _scalr.utils import FlowLogger


class FeatureChunking:
    '''
        FeatureChunking trains models for subsetted datasets, each containing
        `feature_chunksize` features
    '''

    def __init__(self,
                 feature_chunksize: int,
                 chunk_model_config: dict,
                 chunk_model_train_config: dict,
                 train_data: Union[AnnData, AnnCollection],
                 val_data: Union[AnnData, AnnCollection],
                 target: str,
                 mappings: dict,
                 dirpath: str = None,
                 device: str = 'cpu'):
        """
        Args:
            feature_chunksize (int): number of features in one subset
            chunk_model_config (dict): chunked model config
            chunk_model_train_config (dict): chunked model training config
            train_data (Union[AnnData, AnnCollection]): train dataset
            val_data (Union[AnnData, AnnCollection]): validation dataset
            target (str): target to train model
            mappings (dict): mapping of target to labels
            dirpath (str, optional): dirpath to store chunked model weights. Defaults to None.
            device (str, optional): device to train models on. Defaults to 'cpu'.
        """
        self.event_logger = EventLogger('FeatureChunking')

        self.feature_chunksize = feature_chunksize
        self.chunk_model_config = chunk_model_config
        self.chunk_model_train_config = chunk_model_train_config
        self.train_data = train_data
        self.val_data = val_data
        self.target = target
        self.mappings = mappings
        self.dirpath = dirpath
        self.device = device

    def train_chunked_models(self):
        self.event_logger.info('Feature chunked models training')
        models = []
        chunked_models_dirpath = path.join(self.dirpath, 'chunked_models')
        os.makedirs(chunked_models_dirpath, exist_ok=True)

        i = 0
        for start in range(0, len(self.train_data.var_names),
                           self.feature_chunksize):
            self.event_logger.info(f'\nChunk {i}')
            chunk_dirpath = path.join(chunked_models_dirpath, str(i))
            os.makedirs(chunk_dirpath, exist_ok=True)
            i += 1

            train_features_subset = self.train_data[:, start:start +
                                                    self.feature_chunksize]
            val_features_subset = self.val_data[:, start:start +
                                                self.feature_chunksize]

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
        return self.chunk_model_config, self.chunk_model_train_config
