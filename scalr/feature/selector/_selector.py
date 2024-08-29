from typing import Union

from pandas import DataFrame

import scalr
from scalr.utils import build_object


class SelectorBase:
    """Base class for Feature Selector from scores"""

    # Abstract
    def get_feature_list(score_matrix: DataFrame,
                         **kwargs) -> Union[list[str], dict]:
        """Given scores of each feature for each class use an algorithm
        to return top features

        Args:
            score_matrix (DataFrame): score of each feature across all classes
                                      [num_classes X num_features]

        Returns:
            list[str]: list of features
        """
        return

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params"""
        return dict()


def build_selector(selector_config: dict) -> tuple[SelectorBase, dict]:
    """Builder object to get Selector, updated selector_config"""
    return build_object(scalr.feature.selector, selector_config)
