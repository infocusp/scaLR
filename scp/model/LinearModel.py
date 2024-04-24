import torch
from torch import nn

class LinearModel(nn.Module):
    def __init__(self, features, dropout=0):
        """
        Deep Nueral Network model with linear layers.

        Args:
            features: List of layers' feauture size going from input_ft -> out_ft. eg. [22858, 2048, 6] (req)
            dropout: dropout after each layer. Floating point value [0,1) (req)
        """
        super().__init__()
        activation = nn.ReLU(inplace=True)
        dropout = nn.Dropout(dropout, inplace=False)
        self.layers = nn.ModuleList()
        n = len(features)
        for i in range(n):
            if i==n-2:break
            self.layers.append(nn.Linear(features[i], features[i+1]))
            self.layers.append(activation)
            self.layers.append(dropout)

        self.out_layer = nn.Linear(features[n-2], features[n-1])
    
    def forward(self, x):
        """pass input through the network.

            Args:
                x: Tensor, shape [batch_size, self.features[0]]

            Returns:
                output dict containing batched self.features[-1]-dimensional vectors in ['cls_output'] key.
        """
        output = {}
        
        for layer in self.layers:
            x = layer(x)

        output['cls_output'] = self.out_layer(x)
        return output