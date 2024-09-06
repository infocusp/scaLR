"""This file generates heatmaps for top genes of particular class w.r.t same top genes in other classes."""

import os
from typing import Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scalr.analysis import AnalysisBase
from scalr.utils import EventLogger


class Heatmap(AnalysisBase):
    '''Class to generate heatmap of top genes classwise.'''

    def __init__(self,
                 top_n_genes: int = 100,
                 save_plot: bool = True,
                 *args,
                 **kwargs):
        """Initialize class with shap arguments.

        Args:
            top_n_genes: Top N genes for each class/label.
        """

        self.top_n_genes = top_n_genes
        self.save_plot = save_plot

        self.event_logger = EventLogger('Heatmap')

    def generate_analysis(self, score_matrix: pd.DataFrame,
                          top_features: Union[dict, list], dirpath: str,
                          **kwargs) -> None:
        """A function to generate heatmap for top features.

        Args:
            score_matrix: Matrix(class * genes) that contains score of each gene per class.
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
