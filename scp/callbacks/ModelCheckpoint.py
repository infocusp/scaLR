import os
import torch

class ModelCheckpoint:
    """
    Model checkpointing callback at regular intervals and best model
    """
    def __init__(self, filepath, interval, model):
        """
        Args:
            filepath: to store the respective model checkpoints
            interval: regular interval of model checkpointing
        """
        
        self.counter = 0
        self.max_validation_acc = float(0)
        self.interval = int(interval)
        self.filepath = filepath

        self.save_check(model.state_dict(), {}, f'{self.filepath}/checkpoints/model_0.pt')

    def save_check(self, model_state_dict, opt_state_dict, path):
        torch.save({'epoch': self.counter,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': opt_state_dict},
                     path
                )
    
    def __call__(self, model_state_dict, opt_state_dict, validation_acc):
        self.counter+=1
        if validation_acc > self.max_validation_acc:
            self.max_validation_acc = validation_acc
            self.save_check(model_state_dict, opt_state_dict, f'{self.filepath}/best_model/model.pt')
        if self.counter % self.interval==0:
            self.save_check(model_state_dict, opt_state_dict, f'{self.filepath}/checkpoints/model_{self.counter}.pt')
