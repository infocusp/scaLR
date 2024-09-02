import os
from typing import Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scalr.analysis import AnalysisBase
from scalr.utils import EventLogger


class Heatmap(AnalysisBase):
    '''Class to generate Heatmap of top genes classwise.'''

    def __init__(self, top_n_genes: int = 100, *args, **kwargs):
        """Initialize class with shap arguments.

        Args:
            top_n_genes: top N genes for each class/label.
        """

        self.top_n_genes = top_n_genes

        self.event_logger = EventLogger('Heatmap')

    def generate_analysis(self, score_matrix: pd.DataFrame,
                          top_features: Union[dict, list], dirpath: str,
                          **kwargs) -> None:
        """Generate heatmap for top features.

        Args:
            score_matrix: class * genes weights metrix.
            top_features: class wise top genes or list of top features.
            dirpath: path to store the heatmap image.
        """

        self.event_logger.heading2("Heatmap generates.")

        if isinstance(top_features, list):
            self.event_logger.info(
                "Heatmap generate for top genes across all classes.")
            top_features = {"all_class_common": top_features}

        for class_name, genes in top_features.items():
            self.plot_heatmap(score_matrix[genes[:self.top_n_genes]].T,
                              f"{dirpath}/heatmaps", f"{class_name}.svg")

        self.event_logger.info(f"Heatmaps stored at: {dirpath}/heatmaps")

    def plot_heatmap(self, class_genes_weights: pd.DataFrame, dirpath: str,
                     filename: str) -> None:
        """
        Generate a heatmap for top n genes across all classes.

        Args:
            class_genes_weights: genes * classes matrix which contains
                                 shap_value/weights of each gene to class.
            dirpath: path to store the heatmap image.
            filename: heatmap image name.
        """

        os.makedirs(dirpath, exist_ok=True)

        sns.set(rc={'figure.figsize': (9, 12)})
        sns.heatmap(class_genes_weights, vmin=-1e-2, vmax=1e-2)

        plt.tight_layout()

        plt.savefig(os.path.join(dirpath, filename))
        plt.clf()
