import torch
from torch import nn

class CustomLossBase(nn.Module):
    """Base class to implement custom loss functions"""
    
    def forward(self, out, preds):
        # Return Loss
        pass

def build_loss(name, **kwargs):
    # Logging
    return getattr([torch.nn, _scalr.nn.loss], name)(**kwargs)