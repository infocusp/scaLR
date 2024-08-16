from pandas import DataFrame

from _scalr.feature.selector import SelectorBase


class AbsMean(SelectorBase):

    def __init__(self, k: int = 1e6):
        self.k = k

    def get_feature_list(self, score_matrix: DataFrame):
        top_features_list = list(score_matrix.abs().mean().sort_values(
            ascending=False).reset_index()['index'][:self.k])
        return top_features_list
