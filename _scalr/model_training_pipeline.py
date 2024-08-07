import torch
from os import path

from _scalr.nn.model import build_model
from _scalr.nn.loss import build_loss_fn
from _scalr.nn.dataloader import build_dataloader
from _scalr.utils import read_data, load_train_val_data_from_config
from _scalr.nn.callbacks import CallbackExecutor
from _scalr.nn.trainer import SimpleModelTrainer


class ModelTrainingPipeline:

    def __init__(self,
                 dirpath: str,
                 model_config: dict,
                 train_config: dict,
                 data_config: dict,
                 device: str = 'cpu'):
        """Class to get trained model from given configs

        Args:
            dirpath (str): path to store checkpoints and logs of model
            model_config (dict): model config
            train_config (dict): model training config
            data_config (dict): data config
            device (str, optional): device to run model on. Defaults to 'cpu'.
        """
        self.data_config = data_config
        self.train_config = train_config
        self.model_config = model_config
        self.device = device

        self.model = build_model(model_config)
        self.opt = self.build_optimizer(train_config.get('optimizer'))
        self.loss_fn = build_loss_fn(train_config.get('loss_fn'))
        self.callbacks = CallbackExecutor(
            dirpath, train_config.get('callbacks', list()))

        if self.train_config.get('resume_from_checkpoint'):
            self.model.load_weights(
                path.join(self.train_config['resume_from_checkpoint'],
                          'model.pt'))
            self.opt.load_state_dict(
                torch.load(
                    path.join(self.train_config['resume_from_checkpoint'],
                              'model.pt'))['optimizer_state_dict'])

    def build_optimizer(self, opt_config):
        return getattr(torch.optim, opt_config['name'])(
            self.model.parameters(), **opt_config.get('params', dict(lr=1e-3)))

    def load_data_from_config(self):
        """load data from data config"""
        self.train_data, self.val_data = load_train_val_data_from_config(
            self.data_config)

    def set_data(self, train_data, val_data):
        """Useful when you don't use data directly from config, but rather by other
        sources like feature chunking, etc."""
        self.train_data = train_data
        self.val_data = val_data

    # TODO: Keep for future?
    # def update_from_data(self):
    #     """In the case where we need the module objects to access data to make
    #     some changes. We can choose not to call this. But some module objects
    #     may not work!"""

    #     self.model.update_from_data()
    #     self.loss_fn.update_from_data()

    def train(self):
        """Trains the model"""
        trainer = SimpleModelTrainer(self.model, self.opt, self.loss_fn,
                                     self.callbacks, self.device)

        dataloader_config = self.train_config.get('dataloader')
        train_dl = build_dataloader(dataloader_config, self.train_data)
        val_dl = build_dataloader(dataloader_config, self.val_data)
        epochs = self.train_config.get('epochs')
        self.model = trainer.train(epochs, train_dl, val_dl)
