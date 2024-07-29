import os
from os import path
import json
from typing import Callable

import anndata as ad
from anndata import AnnData
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from ..utils import write_data, dump_json, read_data


def generate_train_val_test_split_indices(datapath: str,
                                           split_ratio: list[float],
                                           target: str,
                                           stratify: str = None,
                                           dirpath: str = None) -> dict:
    """Generate a list of indices for train/val/test split of whole dataset

    Args:
        datapath: path to full data
        split_ratio: ratio to split number of samples in
        target: target for classification present in `obs`.
        stratify: optional parameter to stratify the split upon parameter.
        dirpath: dirpath to store generated split in json format

    Returns:
        dict with 'train', 'test' and 'val' indices list.

    """
    
    pass