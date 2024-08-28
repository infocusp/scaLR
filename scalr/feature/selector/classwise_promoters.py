from pandas import DataFrame

from scalr.feature.selector import SelectorBase


class ClasswisePromoters(SelectorBase):

    def __init__(self, k: int = 1e6):
        self.k = k

    def get_feature_list(self, score_matrix: DataFrame):
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
        """Class method to get default params for preprocess_config"""
        return dict(k=int(1e6))
