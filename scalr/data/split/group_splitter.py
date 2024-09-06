from pandas import DataFrame
from sklearn.model_selection import GroupShuffleSplit

from scalr.data.split import StratifiedSplitter


class GroupSplitter(StratifiedSplitter):
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
        super().__init__(split_ratio)
        self.stratify = stratify

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

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params for model_config"""
        return dict(split_ratio=[7, 1, 2], stratify='donor_id')
