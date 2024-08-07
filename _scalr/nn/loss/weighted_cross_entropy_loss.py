from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection

import torch
from torch import nn

from _scalr.nn.loss import CustomLossbase

# TODO: INCOMPLETE


class WeightedCrossEntropyLoss(CustomLossbase):

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def from_data(self, data: Union[AnnCollection, AnnData],
                  targets: list[str]):
        """Use the train data to get ratios of each class. We use these
        ratios to form inverse weights for loss function.

        Args:
            data (Union[AnnCollection, AnnData]): train_data for processing
            targets (list[string]): target columns present in `obs`
        """
        weights = 1 / torch.as_tensor(
            data.obs[targets[0]].value_counts()[id2label], dtype=torch.float32)
        total_sum = weights.sum()
        weights /= total_sum

        self.criterion = nn.CrossEntropyLoss(weight=weights)
