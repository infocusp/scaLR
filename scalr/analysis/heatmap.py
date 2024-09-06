"""This file generates heatmaps for top genes of particular class w.r.t same top genes in other classes."""

import os
from typing import Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scalr.analysis import AnalysisBase
from scalr.utils import EventLogger
from scalr.utils import read_data


class Heatmap(AnalysisBase):
    '''Class to generate a heatmap of top genes classwise.'''

    def __init__(self,
                 top_n_genes: int = 100,
                 save_plot: bool = True,
                 score_matrix_path: str = None,
                 top_features_path: str = None,
                 *args,
                 **kwargs):
        """Initialize class with shap arguments.

        Args:
            top_n_genes: top N genes for each class/label.
            save_plot: Where to save plot or show plot.
            score_matrix_path: path to score matrix.
            top_features_path: path to top features.
        """

        self.top_n_genes = top_n_genes
        self.save_plot = save_plot
        self.score_matrix_path = score_matrix_path
        self.top_features_path = top_features_path

        self.event_logger = EventLogger('Heatmap')

    def generate_analysis(self,
                          dirpath: str,
                          score_matrix: pd.DataFrame = None,
                          top_features: Union[dict, list] = None,
                          **kwargs) -> None:
        """A function to generate heatmap for top features.

        Args:
            score_matrix: Matrix(class * genes) that contains a score of each gene per class.
            top_features: Class-wise top genes or list of top features.
            dirpath: Path to store the heatmap image.
        """

        self.event_logger.heading2("Generating Heatmaps.")

        if isinstance(top_features, list):
            self.event_logger.info(
                "Generating heatmap for the same top genes across all classes as provided"
                " `top_features` is a single list and not top genes per class dict."
            )
            top_features = {"all_class_common": top_features}

        if (score_matrix is None) and (top_features is None):

            if not self.score_matrix_path:
                raise ValueError("score_matrix_path required.")

            if not self.top_features_path:
                raise ValueError("top_features_path required.")

            score_matrix = read_data(self.score_matrix_path)
            top_features = read_data(self.top_features_path)

        for class_name, genes in top_features.items():
            self.plot_heatmap(score_matrix[genes[:self.top_n_genes]].T,
                              f"{dirpath}/heatmaps", class_name)

        self.event_logger.info(f"Heatmaps stored at: {dirpath}/heatmaps")

    def plot_heatmap(self, class_genes_weights: pd.DataFrame, dirpath: str,
                     filename: str) -> None:
        """A function to plot a heatmap for top n genes across all classes.

        Args:
            class_genes_weights: Matrix(genes * classes) which contains
                                 shap_value/weights of each gene to class.
            dirpath: Path to store the heatmap image.
            filename: Heatmap image name.
        """

        os.makedirs(dirpath, exist_ok=True)

        sns.set(rc={'figure.figsize': (9, 12)})
        sns.heatmap(class_genes_weights, vmin=-1e-2, vmax=1e-2)

        plt.tight_layout()
        plt.title(filename)

        if self.save_plot:
            plt.savefig(os.path.join(dirpath, f"{filename}.svg"))
        else:
            plt.show()
        plt.clf()
