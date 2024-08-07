from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from pandas import DataFrame

from _scalr.utils import read_data
from _scalr.data.split import SplitterBase


class StratifiedSplitter(SplitterBase):
    """Generate Stratified split of data into train, validation and test
    sets. Stratification ensures samples having same value for `stratify` 
    column, can not belong to different sets.
    """

    def __init__(self, split_ratio: list[float], stratify: str):
        """
        Args:
            split_ratio (list[float]): ratio to split number of samples in
            stratify (str): column name to metadata the split upon in `obs`
        """
        self.stratify = stratify
        self.split_ratio = split_ratio

    def _split_data_with_stratification(
            self, metadata: DataFrame, target: str,
            test_ratio: float) -> tuple[list[int], list[int]]:
        """Function to split given metadata into a training and testing set.

        Args:
            metadata (DataFrame): dataframe containing all samples to be split
            target (str): target for classification present in `obs`
            test_ratio (float): ratio of samples belonging to the test split
            
        Returns:
            (list(int), list(int)): two lists consisting train and test indices 
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
        """Generate a list of indices for train/val/test split of whole dataset

        Args:
            datapath (str): path to full data
            target (str): target for classification present in `obs`

        Returns:
            dict: 'train', 'val' and 'test' indices list
        """
        if not target:
            raise ValueError('Must provide target for StratifiedSplitter')

        adata = read_data(datapath)
        metadata = adata.obs
        metadata['true_index'] = range(len(metadata))
        n_cls = metadata[target].nunique()

        total_ratio = sum(self.split_ratio)
        train_ratio = self.split_ratio[0] / total_ratio
        val_ratio = self.split_ratio[1] / total_ratio
        test_ratio = self.split_ratio[2] / total_ratio

        # split testing and (train+val) indices
        training_inds, testing_inds = self._split_data_with_stratification(
            metadata, target, test_ratio, self.stratify)

        train_val_data = metadata.iloc[training_inds]
        val_ratio = val_ratio / (val_ratio + train_ratio)

        # get train and val indices, relative to the `train_val_data`
        relative_train_inds, relative_val_inds = self._split_data_with_stratification(
            train_val_data, target, val_ratio, self.stratify)

        # get true_indices relative to entire data
        true_test_inds = testing_inds.tolist()
        true_val_inds = train_val_data.iloc[relative_val_inds]['inds'].tolist()
        true_train_inds = train_val_data.iloc[relative_train_inds][
            'inds'].tolist()

        data_split = {
            'train': true_train_inds,
            'val': true_val_inds,
            'test': true_test_inds
        }

        return data_split
