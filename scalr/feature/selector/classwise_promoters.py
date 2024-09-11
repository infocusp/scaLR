"""This file returns top K promoter features per class."""

from pandas import DataFrame

from scalr.feature.selector import SelectorBase


class ClasswisePromoters(SelectorBase):
    """Class for class-wise promoter feature selector strategy.
    
    Classwise scorer returns a dict for each class, containing the top
    positive scored genes.
    """

    def __init__(self, k: int = 1e6):
        """Initialize required parameters for the selector."""
        self.k = k

    def get_feature_list(self, score_matrix: DataFrame):
        """A function to return top features per class using score matrix
        and selector strategy.

        Args:
            score_matrix (DataFrame): Score of each feature across all classes
                                      [num_classes X num_features].

        Returns:
            list[str]: List of top k features.
        """
        classwise_promoters = dict()
        n_cls = len(score_matrix)

        for i in range(n_cls):
            for i in range(n_cls):
                classwise_promoters[score_matrix.index[i]] = score_matrix.iloc[
                    i, :].sort_values(ascending=False).reset_index(
                    )['index'][:self.k].tolist()

        return classwise_promoters

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params for preprocess_config."""
        return dict(k=int(1e6))
