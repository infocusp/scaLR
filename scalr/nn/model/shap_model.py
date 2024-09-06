"""This file is a implementation of Custom SHAP model."""

from torch import nn


class CustomShapModel(nn.Module):
    """Class for custom model for SHAP."""

    def __init__(self, model, key='cls_output'):
        """Initialize required parameters for SHAP model.

        Args:
            model: Trained model which used for shap calculation.
            key: key from model output dict.
        """
        super().__init__()
        self.model = model
        self.key = key

    def forward(self, x):
        """Pass input through the model and return output.
        
        Args:
            x: Tensor.
        """
        output = self.model(x)

        if isinstance(output, dict):
            output = output[self.key]

        return output
