import os
from os import path

import _scalr
from _scalr.utils import read_data, write_data, write_chunkwise_data, build_object


class SplitterBase:

    # Abstract
    def generate_train_val_test_split_indices(datapath: str, target: str,
                                              **kwargs) -> dict:
        """Generate a list of indices for train/val/test split of whole dataset

        Args:
            datapath (str): path to full data
            target (str): target for classification present in `obs`
            **kwargs: any other params needed for splitting

        Returns:
            dict: 'train', 'val' and 'test' indices list
        """
        pass

    def check_splits(datapath: str, data_splits: dict, target: str):
        """Performs certains checks regarding splits and logs
        the distribution of target classes in each split

        TODO: check for class distribution
        
        Args:
            datapath (str): path to full data
            data_splits (dict): split of 'train', 'val' and 'test' indices.
            target (str): classification target column name in `obs`
        """

        adata = read_data(datapath)
        metadata = adata.obs
        n_cls = metadata[target].nunique()

        train_inds = data_splits['train']
        val_inds = data_splits['val']
        test_inds = data_splits['test']

        # check for classes present in splits
        if len(metadata[target].iloc[train_inds].unique()) != n_cls:
            raise Warning('All classes are not present in Train set')

        if len(metadata[target].iloc[val_inds].unique()) != n_cls:
            raise Warning('All classes are not present in Validation set')

        if len(metadata[target].iloc[test_inds].unique()) != n_cls:
            raise Warning('All classes are not present in Test set')

        # Check for overlapping samples
        assert len(set(train_inds).intersection(
            test_inds)) == 0, "Test and Train sets contain overlapping samples"
        assert len(
            set(val_inds).intersection(train_inds)
        ) == 0, "Validation and Train sets contain overlapping samples"
        assert len(
            set(test_inds).intersection(val_inds)
        ) == 0, "Test and Validation sets contain overlapping samples"

        # LOG
        print('Length of train set: ', len(train_inds))
        print('Distribution of train set: ')
        print(metadata[target].iloc[train_inds].value_counts())
        print()

        print('Length of val set: ', len(val_inds))
        print('Distribution of val set: ')
        print(metadata[target].iloc[val_inds].value_counts())
        print()

        print('Length of test set: ', len(test_inds))
        print('Distribution of test set: ')
        print(metadata[target].iloc[test_inds].value_counts())
        print()

    def write_splits(full_datapath: str, data_split_indices: dict,
                     sample_chunksize: int, dirpath: int) -> dict:
        """Writes the train validation and test splits to disk

        Args:
            full_datapath (str): full datapath of data to be split
            data_split_indices (dict): indices of each split
            sample_chunksize (int): number of samples to be written in one file
            dirpath (int): path/to/dir to write data into

        Returns:
            dict: path of each split
        """
        filepaths = {}

        for split in data_split_indices.keys():
            if sample_chunksize:
                dirpath = path.join(dirpath, split)
                os.makedirs(dirpath, exist_ok=True)

                write_chunkwise_data(full_datapath, sample_chunksize, dirpath,
                                     data_split_indices[split])
                filepaths[f'{split}_datapath'] = filepath
            else:
                full_data = read_data(full_datapath)
                filepath = path.join(dirpath, f'{split}.h5ad')
                write_data(full_data[data_split_indices[split]], filepath)

                filepaths[f'{split}_datapath'] = filepath

        return filepaths


def build_splitter(splitter_config):
    return build_object(_scalr.data.split, splitter_config)
