import os

import _scalr
from _scalr.utils import build_object


class AnalysisBase:

    def __init__(self):
        pass

    def generate_analysis(self, model, test_data, test_dl, **kwargs):
        pass

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params for model_config"""
        return dict()


def build_analyser(analysis_config):
    return build_object(_scalr.nn.analysis, analysis_config)
