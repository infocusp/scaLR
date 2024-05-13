import torch
from torch import nn
from torch import Tensor

class LinearModel(nn.Module):
    """Deep Neural Network model with linear layers."""
    
    def __init__(self, layers:list[int], dropout:float=0, activation:str = 'relu', weights_init_zero:bool=False):
        """
        Args:
            layers: List of layers' feature size going from input_ft -> out_ft. eg. [22858, 2048, 6] (req)
            dropout: dropout after each layer. Floating point value [0,1)
            activation_class: activation function class after each layer
            weights_init_zero: [Bool] to initialize weights of model to zero
        """
        super().__init__()
        if activation == 'relu':
            activation = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError(
            "Activations to be chosen from ['relu']"
        )
        dropout = nn.Dropout(dropout, inplace=False)
        self.layers = nn.ModuleList()
        n = len(layers)
        for i in range(n-2):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            self.layers.append(activation)
            self.layers.append(dropout)

        self.out_layer = nn.Linear(layers[n-2], layers[n-1])

        if weights_init_zero:
            for layer in self.layers:
                torch.nn.init.constant_(layer.weight, 0.0)
                
            torch.nn.init.constant_(self.out_layer.weight, 0.0)
    
    def forward(self, x:Tensor) -> Tensor:
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