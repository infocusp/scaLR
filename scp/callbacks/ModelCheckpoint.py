import os
import torch

class ModelCheckpoint:
    """
    Model checkpointing callback at regular intervals and best model
    """
    def __init__(self, dirpath, checkpoint_interval=5):
        """
        Args:
            dirpath: to store the respective model checkpoints
            interval: regular interval of model checkpointing
        """
        
        self.epoch = 0
        self.max_validation_acc = float(0)
        self.interval = int(checkpoint_interval)
        self.dirpath = dirpath

        os.makedirs(f'{dirpath}/best_model', exist_ok=True)
        if self.interval:
            os.makedirs(f'{dirpath}/checkpoints', exist_ok=True)

    def save_checkpoint(self, model_state_dict, opt_state_dict, path):
        torch.save({'epoch': self.epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': opt_state_dict},
                     path
                )
    
    def __call__(self, model_state_dict, opt_state_dict, validation_acc):
        self.epoch+=1
        if validation_acc > self.max_validation_acc:
            self.max_validation_acc = validation_acc
            self.save_checkpoint(model_state_dict, opt_state_dict, f'{self.dirpath}/best_model/model.pt')
        if self.interval and self.epoch % self.interval==0:
            self.save_checkpoint(model_state_dict, opt_state_dict, f'{self.dirpath}/checkpoints/model_{self.epoch}.pt')
