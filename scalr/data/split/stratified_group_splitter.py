"""This file is an implementation of stratified group splitter."""

from pandas import DataFrame
from sklearn.model_selection import GroupShuffleSplit

from scalr.data.split import SplitterBase
from scalr.utils import read_data


class StratifiedGroupSplitter(SplitterBase):
    """Class for stratified group splitter.

    Generates split of data into train, validation, and test
    sets. Stratification ensures samples have the same value for `stratify`
    column, can not belong to different sets. Also, it ensures every split
    contains samples from each class available in the data.
    """

    def __init__(self, split_ratio: list[float], stratify: str):
        """Initialize splitter with required parameters.

        Args:
            split_ratio (list[float]): ratio to split number of samples in
            stratify (str): column name to metadata the split upon in `obs`
        """
        super().__init__()
        self.stratify = stratify
        self.split_ratio = split_ratio

    def _split_data_with_stratification(
            self, metadata: DataFrame, target: str,
            test_ratio: float) -> tuple[list[int], list[int]]:
        """A function to split given metadata into a training and testing set.

        Args:
            metadata (DataFrame): Dataframe containing all samples to be split.
            target (str): Target for classification present in `obs`.
            test_ratio (float): Ratio of samples belonging to the test split.

        Returns:
            (list(int), list(int)): Two lists consisting of train and test indices.
        """
        splitter = GroupShuffleSplit(test_size=test_ratio,
                                     n_splits=1,
                                     random_state=42)

        train_inds, test_inds = next(
            splitter.split(metadata,
                           metadata[target],
                           groups=metadata[self.stratify]))

        return train_inds, test_inds

    def generate_train_val_test_split_indices(self, datapath: str,
                                              target: str) -> dict:
        """A function to generate a list of indices for train/val/test split of the whole dataset.

        Args:
            datapath (str): Path to full data.
            target (str): Target for classification present in `obs`.

        Returns:
            dict: 'train', 'val' and 'test' indices list.
        """
        if not target:
            raise ValueError('Must provide target for StratifiedGroupSplitter')

        adata = read_data(datapath)
        metadata = adata.obs
        metadata['true_index'] = range(len(metadata))
        n_cls = metadata[target].nunique()

        if n_cls > 2:
            raise ValueError(
                'StratifiedGroupSplitter only works for binary classification.')

        total_ratio = sum(self.split_ratio)
        train_ratio = self.split_ratio[0] / total_ratio
        val_ratio = self.split_ratio[1] / total_ratio
        val_ratio = val_ratio / (val_ratio + train_ratio)
        test_ratio = self.split_ratio[2] / total_ratio

        train_indices = []
        val_indices = []
        test_indices = []

        for label in metadata[target].unique():
            label_metadata = metadata[metadata[target] == label]

            # Split testing and (train+val) indices.
            relative_train_val_inds, relative_test_inds = self._split_data_with_stratification(
                label_metadata, target, test_ratio)

            train_val_data = label_metadata.iloc[relative_train_val_inds]

            # Get train and val indices, relative to the `train_val_data`.
            relative_train_inds, relative_val_inds = self._split_data_with_stratification(
                train_val_data, target, val_ratio)

            # Get true_indices relative to the entire data.
            test_indices.extend(
                label_metadata.iloc[relative_test_inds]['true_index'].tolist())
            val_indices.extend(
                train_val_data.iloc[relative_val_inds]['true_index'].tolist())
            train_indices.extend(
                train_val_data.iloc[relative_train_inds]['true_index'].tolist())

        data_split = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }

        return data_split

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params for model_config."""
        return dict(split_ratio=[7, 1, 2], stratify='donor_id')
