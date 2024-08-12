from copy import deepcopy
import os
from os import path
from time import time

import torch
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from _scalr.nn.callbacks import CallbackExecutor


class TrainerBase:
    """
    Trainer class to train and validate a model from scratch or resume from checkpoint
    """

    def __init__(self,
                 model: Module,
                 opt: Optimizer,
                 loss_fn: Module,
                 callbacks: CallbackExecutor,
                 device: str = 'cpu'):
        """
        Args:
            model (Module): model to train
            opt (Optimizer): optimizer used for learning
            loss_fn (Module): loss function used for training
            callbacks (CallbackExecutor): callback executor object to carry out callbacks
            device (str, optional): device to train the data on (cuda/cpu). Defaults to 'cpu'.
        """
        self.model = model
        self.opt = opt
        self.loss_fn = loss_fn
        self.callbacks = callbacks
        self.device = device

    def train_one_epoch(self, dl: DataLoader) -> tuple[float, float]:
        """Trains one epoch

        Args:
            dl: training dataloader

        Returns:
            Train Loss, Train Accuracy
        """
        self.model.train()
        total_loss = 0
        hits = 0
        total_samples = 0
        for batch in dl:
            x, y = [example.to(self.device)
                    for example in batch[:-1]], batch[-1].to(self.device)

            out = self.model(*x)['cls_output']
            loss = self.loss_fn(out, y)

            #training
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            #logging
            total_loss += loss.item() * x[0].size(0)
            total_samples += x[0].size(0)
            hits += (torch.argmax(out, dim=1) == y).sum().item()

        total_loss /= total_samples
        accuracy = hits / total_samples
        return total_loss, accuracy

    def validation(self, dl: DataLoader) -> tuple[float, float]:
        """ Validation of data

        Args:
            dl: validation dataloader

        Returns:
            Validation Loss, Validation Accuracy
        """
        self.model.eval()
        total_loss = 0
        hits = 0
        total_samples = 0
        for batch in dl:
            with torch.no_grad():
                x, y = [example.to(self.device)
                        for example in batch[:-1]], batch[-1].to(self.device)
                out = self.model(*x)['cls_output']
                loss = self.loss_fn(out, y)

            #logging
            hits += (torch.argmax(out, dim=1) == y).sum().item()
            total_loss += loss.item() * x[0].size(0)
            total_samples += x[0].size(0)

        total_loss /= total_samples
        accuracy = hits / total_samples

        return total_loss, accuracy

    def train(self, epochs: int, train_dl: DataLoader, val_dl: DataLoader):
        """Trains the model.

        Args:
            epochs: max number of epochs to train model on
            train_dl: training dataloader
            val_dl: validation dataloader
        """
        best_val_acc = 0

        for epoch in range(epochs):
            ep_start = time()
            print(f'Epoch {epoch+1}:')
            train_loss, train_acc = self.train_one_epoch(train_dl)
            print(
                f'Training Loss: {train_loss} || Training Accuracy: {train_acc}'
            )
            val_loss, val_acc = self.validation(val_dl)
            print(
                f'Validation Loss: {val_loss} || Validation Accuracy: {val_acc}'
            )
            ep_end = time()
            print(f'Time: {ep_end-ep_start}\n', flush=True)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = deepcopy(self.model)

            if self.callbacks.execute(model_state_dict=self.model.state_dict(),
                                      opt_state_dict=self.opt.state_dict(),
                                      train_loss=train_loss,
                                      train_acc=train_acc,
                                      val_loss=val_loss,
                                      val_acc=val_acc):
                break

        return best_model
