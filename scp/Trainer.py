import os
import torch
from torch import nn
from torch.utils.data import Dataset, IterableDataset, DataLoader
from .callbacks import CallBack
from time import time
from .tokenizer import tokenize_and_pad_batch


class Trainer:
    """
    Trainer class to train and validate a model from scratch or resume from checkpoint
    """
    def __init__(self, model, opt, lr, l2, loss_fn, callbacks : dict, device='cuda', filepath='.', model_checkpoint=None):
        """
        Args:
            model: model to train
            opt: optimizer to train model parameters
            lr: learning rate for optimizer
            l2: L2 penalty for weights
            loss_fn: loss function for training
            callbacks: callback params : dict {'model_checkpoint_interval', 'early_stop_patience', 'early_stop_min_delta'}
            device: device for compuations ('cpu'/'cuda')
            filepath: filepath for storing logs, checkpoints, best_model
            model_checkpoint: path to resume training from given checkpoint
        """
        self.device = device
        if not torch.cuda.is_available(): self.device='cpu'

        self.model = model.to(self.device)
        self.opt = opt(self.model.parameters(), lr=lr, weight_decay=l2)

        if model_checkpoint is not None:
            state_dict = torch.load(f'{model_checkpoint}/model.pt')
            self.model.load_state_dict(state_dict['model_state_dict'])
            self.opt.load_state_dict(state_dict['optimizer_state_dict'])
        
        self.loss_fn = loss_fn
        self.filepath = filepath
        self.callbacks = callbacks

    def train_one_epoch(self, dl):
        """ training one epoch """
        self.model.train()
        total_loss = 0
        hits = 0
        total_samples = 0
        for batch in dl:
            x, y = [x_.to(self.device) for x_ in batch[:-1]], batch[-1].to(self.device)
            
            out = self.model(*x)['cls_output']
            loss = self.loss_fn(out, y)
    
            #training
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
    
            #logging
            total_loss += loss.item()*x[0].size(0)
            total_samples += x[0].size(0)
            hits += (torch.argmax(out, dim=1)==y).sum().item()

        total_loss/=total_samples
        accuracy = hits/total_samples
        return total_loss, accuracy

    def validation(self, dl):
        """ validation after training one epoch """
        self.model.eval()
        total_loss = 0
        hits = 0
        total_samples = 0
        for batch in dl:
            with torch.no_grad():
                x, y = [x_.to(self.device) for x_ in batch[:-1]], batch[-1].to(self.device)
                out = self.model(*x)['cls_output']
                loss = self.loss_fn(out, y)
    
            #logging
            hits += (torch.argmax(out, dim=1)==y).sum().item()
            total_loss += loss.item()*x[0].size(0)
            total_samples += x[0].size(0)
    
        total_loss/=total_samples
        accuracy = hits/total_samples
    
        return total_loss, accuracy

    def train(self, epochs, train_dl, val_dl):
        """train function

        Args:
            epochs: max number of epochs to train model on
            train_dl: training dataloader
            val_dl: validation dataloader
        """
    
        callback = CallBack(filepath=self.filepath, callbacks=self.callbacks)
        
        for epoch in range(epochs):
            ep_start = time()
            print(f'Epoch {epoch+1}:')
            train_loss, train_acc = self.train_one_epoch(train_dl)
            print(f'Training Loss: {train_loss} || Training Accuracy: {train_acc}')
            val_loss, val_acc = self.validation(val_dl)
            print(f'Validation Loss: {val_loss} || Validation Accuracy: {val_acc}')
            ep_end = time()
            print(f'Time: {ep_end-ep_start}\n')
            
            if callback(self.model.state_dict(), self.opt.state_dict(), train_loss, train_acc, val_loss, val_acc): break

        self.model.load_state_dict(torch.load(f'{self.filepath}/best_model/model.pt')['model_state_dict'])
        torch.save(self.model, f'{self.filepath}/best_model/model.bin')
    
        return















