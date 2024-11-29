"""This file is a base class for splitter."""

import os
from os import path
from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection

import scalr
from scalr.utils import build_object
from scalr.utils import EventLogger
from scalr.utils import read_data
from scalr.utils import write_chunkwise_data
from scalr.utils import write_data


class SplitterBase:
    """Base class for splitter, to make Train|Val|Test Splits."""

    def __init__(self):
        self.event_logger = EventLogger('Splitter')

    # Abstract
    def generate_train_val_test_split_indices(datapath: str, target: str,
                                              **kwargs) -> dict:
        """Generate a list of indices for train/val/test split of whole dataset.

        Args:
            datapath (str): Path to full data.
            target (str): Target for classification present in `obs`.
            **kwargs: Any other params needed for splitting.

        Returns:
            dict: 'train', 'val' and 'test' indices list.
        """
        pass

    def check_splits(self, datapath: str, data_splits: dict, target: str):
        """This function performs certain checks regarding splits and logs
        the distribution of target classes in each split.

        Args:
            datapath (str): Path to full data.
            data_splits (dict): Split of 'train', 'val' and 'test' indices.
            target (str): Classification target column name in `obs`.
        """

        adata = read_data(datapath)
        metadata = adata.obs
        n_cls = metadata[target].nunique()

        train_inds = data_splits['train']
        val_inds = data_splits['val']
        test_inds = data_splits['test']

        # Check for classes present in splits.
        if len(metadata[target].iloc[train_inds].unique()) != n_cls:
            self.event_logger.warning(
                'All classes are not present in Train set')

        if len(metadata[target].iloc[val_inds].unique()) != n_cls:
            self.event_logger.warning(
                'All classes are not present in Validation set')

        if len(metadata[target].iloc[test_inds].unique()) != n_cls:
            self.event_logger.warning('All classes are not present in Test set')

        # Check for overlapping samples.
        assert len(set(train_inds).intersection(
            test_inds)) == 0, "Test and Train sets contain overlapping samples"
        assert len(
            set(val_inds).intersection(train_inds)
        ) == 0, "Validation and Train sets contain overlapping samples"
        assert len(set(test_inds).intersection(val_inds)
                  ) == 0, "Test and Validation sets contain overlapping samples"

        # LOGGING.
        self.event_logger.info('Train|Validation|Test Splits\n')
        self.event_logger.info(f'Length of train set: {len(train_inds)}')
        self.event_logger.info(f'Distribution of train set: ')
        self.event_logger.info(
            f'{metadata[target].iloc[train_inds].value_counts()}\n')

        self.event_logger.info(f'Length of val set: {len(val_inds)}')
        self.event_logger.info(f'Distribution of val set: ')
        self.event_logger.info(
            f'{metadata[target].iloc[val_inds].value_counts()}\n')

        self.event_logger.info(f'Length of test set: {len(test_inds)}')
        self.event_logger.info(f'Distribution of test set: ')
        self.event_logger.info(
            f'{metadata[target].iloc[test_inds].value_counts()}\n')

    def write_splits(self,
                     full_data: Union[AnnData, AnnCollection],
                     data_split_indices: dict,
                     sample_chunksize: int,
                     dirpath: int,
                     num_workers: int = None):
        """THis function writes the train validation and test splits to the disk.

        Args:
            full_data (Union[AnnData, AnnCollection]): Full data to be split.
            data_split_indices (dict): Indices of each split.
            sample_chunksize (int): Number of samples to be written in one file.
            dirpath (int): Path to write data into.
            num_workers (int): number of jobs to run in parallel for data writing.
        """

        for split in data_split_indices.keys():
            if sample_chunksize:
                split_dirpath = path.join(dirpath, split)
                os.makedirs(split_dirpath, exist_ok=True)
                write_chunkwise_data(full_data,
                                     sample_chunksize,
                                     split_dirpath,
                                     data_split_indices[split],
                                     num_workers=num_workers)
            else:
                filepath = path.join(dirpath, f'{split}.h5ad')
                write_data(full_data[data_split_indices[split]].to_memory(),
                           filepath)

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params for model_config."""
        return dict()


def build_splitter(splitter_config: dict) -> tuple[SplitterBase, dict]:
    """Builder object to get splitter, updated splitter_config."""
    return build_object(scalr.data.split, splitter_config)
