import torch
from torch import nn

class LinearModel(nn.Module):
    def __init__(self, layers, dropout=0, weights_init_zero=False):
        """
        Deep Nueral Network model with linear layers.

        Args:
            layers: List of layers' feauture size going from input_ft -> out_ft. eg. [22858, 2048, 6] (req)
            dropout: dropout after each layer. Floating point value [0,1) (req)
        """
        super().__init__()
        activation = nn.ReLU(inplace=True)
        dropout = nn.Dropout(dropout, inplace=False)
        self.layers = nn.ModuleList()
        n = len(layers)
        for i in range(n):
            if i==n-2:break
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            self.layers.append(activation)
            self.layers.append(dropout)

        self.out_layer = nn.Linear(layers[n-2], layers[n-1])

        if weights_init_zero:
            for layer in self.layers:
                torch.nn.init.constant_(layer.weight, 0.0)
                
            torch.nn.init.constant_(self.out_layer.weight, 0.0)
    
    def forward(self, x):
        """pass input through the network.

            Args:
                x: Tensor, shape [batch_size, layers[0]]

            Returns:
                output dict containing batched layers[-1]-dimensional vectors in ['cls_output'] key.
        """
        output = {}
        
        for layer in self.layers:
            x = layer(x)

        output['cls_output'] = self.out_layer(x)
        return output