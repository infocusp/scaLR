import torch
from torch.utils.tensorboard import SummaryWriter

class Logs:
    """
    Tensorboard logging of training process
    """
    def __init__(self, filepath):
        """
        Args:
            filepath: to store the experiment logs
        """
        self.writer = SummaryWriter(filepath+'/logs')
        self.epoch = 0
    
    def __call__(self,train_loss, train_acc, val_loss, val_acc):
        """
        logging: train_loss, val_loss, train_accuracy, val_accuracy
        """
        self.epoch+=1
        self.writer.add_scalar('Loss/train', train_loss, self.epoch)
        self.writer.add_scalar('Loss/val', val_loss, self.epoch)
        self.writer.add_scalar('Accuracy/train', train_acc, self.epoch)
        self.writer.add_scalar('Accuracy/val', val_acc, self.epoch)