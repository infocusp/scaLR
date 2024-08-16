from typing import Union

from pandas import DataFrame

import _scalr
from _scalr.utils import build_object


class SelectorBase:

    def get_classwise_feature_list(score_matrix: DataFrame) -> dict:
        """Optional method to return dict of classwise important features
        list.
        """
        return

    def get_feature_list(score_matrix: DataFrame, **kwargs) -> list[str]:
        """Given scores of each feature for each class use an algorithm
        to return top features"""
        return

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params"""
        return dict()


def build_selector(selector_config):
    return build_object(_scalr.feature.selector, selector_config)
