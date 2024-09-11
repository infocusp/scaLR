"""This file generates gene recall curves for reference genes in provided models ranked genes"""

import json
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scalr.analysis import AnalysisBase
from scalr.feature.selector import build_selector
from scalr.utils import EventLogger
from scalr.utils import read_data


class GeneRecallCurve(AnalysisBase):
    '''Class to generate gene recall curve.'''

    def __init__(self,
                 reference_genes_path: str,
                 ranked_genes_path_dict: dict = None,
                 top_K: int = 150,
                 plots_per_row: int = 3,
                 save_plots: bool = True,
                 features_selector: dict = None):
        '''Load required parameters for gene recall curve.

        Args:
            reference_genes_path: Reference genes csv path.
            ranked_genes_path: List of ranked genes csvs we want to check recall against. The list is used to support
                               plotting gene recall for multiple models at once.
            top_K: Top K genes to consider from ranked genes to check gene recall.
            plots_per_row: Number of categories to plot gene recall per row.
            save_plots: Flag indicating whether to store gene recall curves or not.
            feature_selector: Feature selection config containing aggregation strategy for `score_matrix`.
        '''
        self.ranked_genes_path_dict = ranked_genes_path_dict
        self.reference_genes_path = reference_genes_path
        self.top_K = top_K
        self.plots_per_row = plots_per_row
        self.save_plots = save_plots

        # Build feature selector if `score_matrix` is to be used to generate gene recall curve.
        if features_selector:
            self.selector, _ = build_selector(features_selector)

        self.event_logger = EventLogger('Gene Recall Curve analysis')

    def generate_analysis(self,
                          score_matrix: pd.DataFrame = None,
                          dirpath: str = '.',
                          **kwargs):
        '''This function calls function to generate gene recall after setting a few parameters.

        Args:
            score_matrix: Matrix that contains a score of each gene for each category.
            dirpath: Path to store gene recall curve if applicable.
        '''

        self.event_logger.heading2("Gene Recall Curve analysis")

        # Handle to ensure reference genes are provided.
        if not self.reference_genes_path:
            raise ValueError(
                'Reference genes are required for generating gene recall curves!'
            )
        reference_genes_df = read_data(self.reference_genes_path, index_col=0)

        # Extracting ranked genes.
        if not self.ranked_genes_path_dict:
            try:
                ranked_genes_df_dict = {
                    'pipeline_model':
                        pd.DataFrame.from_dict(
                            self.selector.get_feature_list(
                                score_matrix=score_matrix))
                }
            except:
                raise ValueError(
                    'There is some issue in `score_matrix` that is generated during the pipeline run. Please check!'
                )
        else:
            ranked_genes_df_dict = {}
            for model_name, genes_path in self.ranked_genes_path_dict.items():
                ranked_genes_df_dict[model_name] = read_data(genes_path,
                                                             index_col=0)

        # Generate gene recall curve.
        self.plot_gene_recall(ranked_genes_df_dict=ranked_genes_df_dict,
                              reference_genes_df=reference_genes_df,
                              dirpath=dirpath)

    def plot_gene_recall(self,
                         ranked_genes_df_dict: dict,
                         reference_genes_df: pd.DataFrame,
                         dirpath: str = '.',
                         title: str = ''):
        """This function plots & stores the gene recall curve for reference genes in provided ranked genes.

        It also stores the reference genes along with their ranks for each model in a json file for further
        analysis to the user.

        Args:
            ranked_genes_df_dict: Pipeline generated ranked genes dataframe.
            reference_genes_df: Reference genes dataframe.
            top_K: The top K-ranked genes in which reference genes are to be looked for.
            dirpath: Path to store gene recall plot and json.
            plot_type: Type of gene recall - per category or aggregated across all categories.
        """

        # Common categories across reference and ranked genes dataframes.
        common_categories = reference_genes_df.columns
        for model in ranked_genes_df_dict.values():
            common_categories = set(common_categories).intersection(
                model.columns)
        n_plots = len(common_categories)
        self.event_logger.info(
            f'\n{n_plots} categories matches between ranked genes & reference genes dataframes, namely: {common_categories}'
        )
        n_cols = min(self.plots_per_row, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axs = plt.subplots(n_rows,
                                n_cols,
                                figsize=(n_cols * self.plots_per_row,
                                         n_rows * self.plots_per_row),
                                squeeze=False)
        axs = axs.flatten()
        fig.suptitle(f'Recall of reference genes w.r.t ranked genes - {title}')

        # Plotting baseline gene recall curve.
        for i, category in enumerate(common_categories):
            ranked_genes = list(
                ranked_genes_df_dict.values())[0][category].values
            ref_genes = reference_genes_df[category].dropna().values
            k = self.top_K
            if k < len(ref_genes):
                k = len(ref_genes)

            if not len(ranked_genes) or not len(ref_genes):
                raise Exception(
                    'Ranked genes or ref genes list cannot be empty.')

            # Adjusting k if expected k > number of ranked genes.
            k = min(k, len(ranked_genes))

            # Building baseline curve.
            step = k // len(ref_genes)
            baseline = [i // step for i in range(1, 1 + k)]

            axs[i].plot(list(range(1, 1 + k)), baseline, label='baseline')
            axs[i].legend()

        # Plotting gene recall for defined ranked genes dataframes.
        gene_recall_dict = {}
        for df_name, ranked_genes_df in ranked_genes_df_dict.items():
            gene_recall_dict[df_name] = {}
            for i, category in enumerate(common_categories):
                ranked_genes = ranked_genes_df[category].values
                ref_genes = reference_genes_df[category].dropna().values
                k = self.top_K

                self.event_logger.info(f'\nCategory - {category}')
                # Removing reference genes that are not available in the list of the ranked genes.
                self.event_logger.info(
                    f'-- Number of reference genes provided : {len(ref_genes)}')
                ref_genes = list(set(ref_genes).intersection(ranked_genes))
                self.event_logger.info(
                    f'-- Number of reference genes found in ranked genes : {len(ref_genes)}'
                )

                if k < len(ref_genes):
                    self.event_logger.info(
                        f'-- top_K={k} should be greater than or equal to #reference_genes={len(ref_genes)}, setting'
                        f' top_K= #reference_genes({len(ref_genes)})')
                    k = len(ref_genes)

                if not len(ranked_genes) or not len(ref_genes):
                    raise Exception(
                        'Ranked genes or ref genes list cannot be empty.')

                # Adjusting k if expected k > number of ranked genes.
                if k > len(ranked_genes):
                    self.event_logger.info(
                        f'-- Setting k={len(ranked_genes)} as #ranked_genes < {k}'
                    )
                    k = len(ranked_genes)

                order_in_lit = {}
                points = []

                count = 0
                for rank, gene in enumerate(ranked_genes[:k]):
                    if gene in ref_genes:
                        count += 1
                        order_in_lit[rank] = gene
                    points.append(count)

                axs[i].plot(list(range(1, 1 + k)), points, label=df_name)
                axs[i].set_title(category)
                axs[i].set_xlabel('# Top-ranked genes (k)')
                axs[i].set_ylabel('# Reference genes')
                axs[i].legend()

                gene_recall_dict[df_name][category] = order_in_lit

        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        if self.save_plots:
            plt.savefig(path.join(dirpath, 'gene_recall_curve.svg'))
            with open(path.join(dirpath, 'gene_recall_curve_info.json'),
                      'w') as f:
                json.dump(gene_recall_dict, f, indent=6)
            self.event_logger.info(
                f'\nGene recall curves stored at path: `{path.join(dirpath, "gene_recall_curve.svg")}`'
            )
        else:
            plt.show()

        plt.close()

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params for preprocess_config."""
        return dict(ranked_genes_path_dict={},
                    top_K=150,
                    plots_per_row=3,
                    save_plots=True,
                    features_selector={})
