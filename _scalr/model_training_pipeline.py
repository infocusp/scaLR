from copy import deepcopy
import os
from os import path
from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import torch

import _scalr
from _scalr.nn.callbacks import CallbackExecutor
from _scalr.nn.dataloader import build_dataloader
from _scalr.nn.loss import build_loss_fn
from _scalr.nn.model import build_model
from _scalr.utils import load_train_val_data_from_config
from _scalr.utils import read_data
from _scalr.utils import write_data


class ModelTrainingPipeline:

    def __init__(self,
                 model_config: dict,
                 train_config: dict,
                 dirpath: str = None,
                 device: str = 'cpu'):
        """Class to get trained model from given configs

        Args:
            dirpath (str): path to store checkpoints and logs of model
            model_config (dict): model config
            train_config (dict): model training config
            device (str, optional): device to run model on. Defaults to 'cpu'.
        """
        self.train_config = train_config
        self.model_config = model_config
        self.device = device
        self.dirpath = dirpath

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

    def build_model_training_artifacts(self):
        # Model Building
        self.model, self.model_config = build_model(self.model_config)
        self.model.to(self.device)

        # Optimizer Building
        opt_config = deepcopy(self.train_config.get('optimizer'))
        self.opt, opt_config = self.build_optimizer(
            self.train_config.get('optimizer'))
        self.train_config['optimizer'] = opt_config

        # Build Loss Function
        self.loss_fn, loss_config = build_loss_fn(
            deepcopy(self.train_config.get('loss', dict())))
        self.train_config['loss'] = loss_config
        self.loss_fn.to(self.device)

        # Build Callbacks executor
        self.callbacks = CallbackExecutor(
            self.dirpath, self.train_config.get('callbacks', list()))

        # Resuming from checkpoint
        # QUESTION:
        # Do we want to make model according to checkpoint?
        # OR
        # Only load weights from a checkpoint?
        if self.train_config.get('resume_from_model_weights'):
            self.model.load_weights(
                path.join(self.train_config['resume_from_model_weights'],
                          'model.pt'))
            self.opt.load_state_dict(
                torch.load(
                    path.join(self.train_config['resume_from_model_weights'],
                              'model.pt'))['optimizer_state_dict'])

    def build_optimizer(self, opt_config: dict = None):
        if not opt_config:
            opt_config = dict()
        name = opt_config.get('name', 'Adam')
        opt_config['name'] = name
        params = opt_config.get('params', dict(lr=1e-3))
        opt_config['params'] = params

        opt = getattr(torch.optim, name)(self.model.parameters(), **params)
        return opt, opt_config

    def train(self):
        """Trains the model"""
        # Build Trainer
        trainer_name = self.train_config.get('trainer', 'SimpleModelTrainer')
        self.train_config['trainer'] = trainer_name

        Trainer = getattr(_scalr.nn.trainer, trainer_name)
        trainer = Trainer(self.model, self.opt, self.loss_fn, self.callbacks,
                          self.device)

        # Build DataLoaders
        dataloader_config = self.train_config.get('dataloader')
        train_dl, dataloader_config = build_dataloader(dataloader_config,
                                                       self.train_data,
                                                       self.target,
                                                       self.mappings)
        val_dl, dataloader_config = build_dataloader(dataloader_config,
                                                     self.val_data, self.target,
                                                     self.mappings)
        self.train_config['dataloader'] = dataloader_config

        epochs = self.train_config.get('epochs', 1)
        self.train_config['epochs'] = epochs

        # Train and store the best model
        best_model = trainer.train(epochs, train_dl, val_dl)
        if self.dirpath:
            best_model_dir = path.join(self.dirpath, 'best_model')
            os.makedirs(best_model_dir, exist_ok=True)
            best_model.save_weights(path.join(best_model_dir, 'model.pt'))
            write_data(self.model_config,
                       path.join(best_model_dir, 'model_config.yaml'))
            write_data(self.mappings, path.join(best_model_dir,
                                                'mappings.json'))

        return best_model

    def get_updated_config(self):
        """Returns updated configs
        """
        return self.model_config, self.train_config
