from pandas import DataFrame

from scalr.feature.selector import SelectorBase


class AbsMean(SelectorBase):
    """Use the absolute mean across all classes as the score of
    the feature"""

    def __init__(self, k: int = 1e6):
        self.k = k

    def get_feature_list(self, score_matrix: DataFrame) -> list[str]:
        """
        Args:
            score_matrix (DataFrame): score of each feature across all classes
                                      [num_classes X num_features]

        Returns:
            list[str]: list of top k features
        """
        top_features_list = list(score_matrix.abs().mean().sort_values(
            ascending=False).reset_index()['index'][:self.k])
        return top_features_list

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params for preprocess_config"""
        return dict(k=int(1e6))
