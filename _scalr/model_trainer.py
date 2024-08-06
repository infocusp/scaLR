import torch

from _scalr.nn.trainer import build_trainer
from _scalr.nn.model import build_model
from _scalr.nn.loss import build_loss
from _scalr.nn.callbacks import CallbackExecutor
from _scalr.nn.dataloader import build_dataloader

class ModelTrainer:
    
    def __init__(self, model_config, train_config, data_config):
        # Load params here
        
        # Initialize the model, optimizer, loss and callbacks class objects here
        # Initialize the weights (in case of resume from checkpointing) for model and opt 
        # Finally create a trainer object for model training
        
        self.opt = self.build_optimizer
        self.loss_fn = build_loss
        self.model = build_model(model_config)
        self.callback_executor = CallbackExecutor
        self.trainer = build_trainer(trainer_name, model, opt, loss, callbacks)
        
        # build_dataloaders
        
        pass

    def build_optimizer(name):
        return getattr(torch.optim, name)
    
    def train(self):
        """Trains the model.

        Args:
            epochs: max number of epochs to train model on
            train_dl: training dataloader
            val_dl: validation dataloader
        """

        for epoch in range(epochs):
            train_one_epoch()
            validation()
            callback()
            
        
    def get_model_weights():
        return
