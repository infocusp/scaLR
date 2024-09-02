"""This file returns top K features(can be promoters or inhibitors as well) per category."""
from pandas import DataFrame

from scalr.feature.selector import SelectorBase


class ClasswiseAbs(SelectorBase):
    """Classwise scorer returns a dict for each class, containing top
    absolute scores of genes"""

    def __init__(self, k: int = 1e6) -> dict:
        self.k = k

    def get_feature_list(self, score_matrix: DataFrame):
        """
        Args:
            score_matrix (DataFrame): score of each feature across all classes
                                      [num_classes X num_features]

        Returns:
            dict: list of top_k features for each class
        """
        classwise_abs = dict()
        n_cls = len(score_matrix)

        for i in range(n_cls):
            for i in range(n_cls):
                classwise_abs[score_matrix.index[i]] = abs(
                    score_matrix.iloc[i, :]).sort_values(
                        ascending=False).reset_index()['index'][:self.k].tolist(
                        )

        return classwise_abs

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params for preprocess_config"""
        return dict(k=int(1e6))
