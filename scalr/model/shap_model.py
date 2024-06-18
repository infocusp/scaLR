from torch import nn


class CustomShapModel(nn.Module):
    """Custom model for SHAP."""

    def __init__(self, model, key='cls_output'):
        """
        Args:
            model: Trained model which used for shap calculation.
            key: key from model output dict.
        """
        super().__init__()
        self.model = model
        self.key = key

    def forward(self, x):
        return self.model(x)[self.key]
