"""This file generates differential gene expression analysis using Pseudobulk approach and stores the results."""

import os
from os import path
from typing import Optional, Tuple, Union

from anndata import AnnData
import anndata as ad
from anndata.experimental import AnnCollection
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import scanpy as sc

from scalr import utils
from scalr.analysis import AnalysisBase
from scalr.utils import EventLogger
from scalr.utils import read_data


class DgePseudoBulk(AnalysisBase):
    '''Class to perform differential gene expression analysis using Pseudobulk approach.'''

    def __init__(self,
                 celltype_column: str,
                 design_factor: str,
                 factor_categories: list[str],
                 sum_column: str,
                 cell_subsets: list[str] = None,
                 min_cell_threshold: int = 1,
                 fold_change: Union[float, int] = 1.5,
                 p_val: Union[float, int] = 0.05,
                 y_lim_tuple: Optional[Tuple[float, ...]] = None,
                 save_plot: bool = True,
                 stdout: bool = False):
        '''DgePseudoBulk parameters initialization.
        
        Args: 
            celltype_column: Column name in `anndata.obs` containing all the cell types.
            design_factor: Column name in `anndata.obs` containing different factor levels or categories for
                           differential gene expression analysis.         
            factor_categories: List of conditions in `design_factor` to make design matrix for.
            sum_column: Column name to sum values across samples.
            cell_subsets: Selcted list of cells in 'celltype_column' to subset the anndata.
            min_cell_threshold: Minimum number of subjects with aggregated nonzero gene expression values for a gene.
                                Used for filtering noisy genes.
            fold_change: Fold change to filter the differentially expressed genes for the volcano plot.
            p_val: p value, to filter the differentially expressed genes for the volcano plot.
            y_lim_tuple: Values to adjust the Y-axis limits of the plot.
            save_plot: Boolean value to save plot.
            stdout : Flag to print logs to stdout.
        '''

        self.celltype_column = celltype_column
        self.design_factor = design_factor
        # Hacky fix since pydeseq2 does not support `_` in `design_factor` or factor levels.
        self.design_factor_no_undrscr = design_factor.replace('_', '')
        self.factor_categories = [
            factor.replace('_', '-') for factor in factor_categories
        ]
        self.sum_column = sum_column
        self.cell_subsets = cell_subsets
        self.min_cell_threshold = min_cell_threshold
        self.fold_change = fold_change
        self.p_val = p_val
        self.y_lim_tuple = y_lim_tuple
        self.save_plot = save_plot
        self.stdout = stdout

    def _make_design_matrix(self, adata: AnnData, cell_type: str):
        '''Method to subset an anndata as per a cell type and 
        make design matrix based upon the factor levels in design_factor.
    
        Args:
            adata: AnnData.
            cell_type: Cell type to subset data on, belonging to `celltype_column`.
    
        Returns:
            AnnData oject of design matrix.
        '''

        if isinstance(adata, AnnData):
            adata = AnnCollection([adata])

        design_matrix_list = []

        fix_data = adata[adata.obs[self.celltype_column] == cell_type]
        for condition in self.factor_categories:
            condition_subset = fix_data[fix_data.obs[self.design_factor] ==
                                        condition]
            for sum_sample in condition_subset.obs[self.sum_column].unique():
                sum_subset = condition_subset[condition_subset.obs[
                    self.sum_column] == sum_sample]
                subdata = ad.AnnData(
                    X=sum_subset[:].X.sum(axis=0).reshape(
                        1, len(sum_subset.var_names)),
                    var=DataFrame(index=sum_subset.var_names),
                    obs=DataFrame(index=[f'{sum_sample}_{condition}']))
                subdata.obs[self.design_factor_no_undrscr] = [condition]
                design_matrix_list.append(subdata)

        design_matrix = ad.concat(design_matrix_list)
        return design_matrix

    def get_differential_expression_results(self, design_matrix: AnnData,
                                            cell_type: str, dirpath: str):
        '''Method to get differential gene expression analysis results.
    
        Args:
            design_matrix: AnnData generated using '_make_design_matrix'.
            cell_type: Cell type used to subset the input anndata.
            dirpath: Path to save the result.
    
        Returns:
            A pandas DataFrame object containing differential gene expression results.
        '''

        count_matrix = design_matrix.to_df()
        count_matrix_int = count_matrix.round().astype(int)
        metadata_ad = design_matrix.obs

        deseq_dataset = DeseqDataSet(
            counts=count_matrix_int,
            metadata=metadata_ad,
            design_factors=self.design_factor_no_undrscr)

        # removing genes with zero expression across all the sample
        sc.pp.filter_genes(deseq_dataset, min_cells=self.min_cell_threshold)
        deseq_dataset.deseq2()

        result_stats = DeseqStats(deseq_dataset,
                                  contrast=(self.design_factor_no_undrscr,
                                            *self.factor_categories))
        result_stats.summary()
        results_df = result_stats.results_df
        results_df.index.name = 'gene'

        if dirpath:
            _cell_type = cell_type.replace(" ", "")
            factor_0 = (self.factor_categories[0]).replace(" ", "")
            factor_1 = (self.factor_categories[1]).replace(" ", "")
            results_df.to_csv(
                path.join(dirpath,
                          f'pbkDGE_{_cell_type}_{factor_0}_vs_{factor_1}.csv'))
        return results_df

    def plot_volcano(self, dge_results_df: DataFrame, cell_type: str,
                     dirpath: str):
        '''Method to generate volcano plot of differential gene expression results 
        and store it on disk.
    
        Args:
            dge_results_df: Differential gene expression results in dataframe.
            cell_type: Cell type used to subset the input anndata.
            dirpath: Path to save the result.
        '''

        log2_fold_chnage = np.log2(self.fold_change)
        neg_log10_pval = -np.log10(self.p_val)
        dge_results_df['-log10(pvalue)'] = -np.log10(dge_results_df['pvalue'])

        upregulated_gene = (dge_results_df['log2FoldChange'] >= log2_fold_chnage
                           ) & (dge_results_df['-log10(pvalue)']
                                >= (neg_log10_pval))
        downregulated_gene = (dge_results_df['log2FoldChange'] <= (
            -log2_fold_chnage)) & (dge_results_df['-log10(pvalue)']
                                   >= (neg_log10_pval))

        unsignificant_gene = dge_results_df['-log10(pvalue)'] <= (
            neg_log10_pval)
        signi_genes_between_up_n_down = ~(upregulated_gene | downregulated_gene
                                          | unsignificant_gene)

        plt.figure(figsize=(10, 6))
        plt.scatter(dge_results_df.loc[upregulated_gene, 'log2FoldChange'],
                    dge_results_df.loc[upregulated_gene, '-log10(pvalue)'],
                    color='red',
                    alpha=0.2,
                    s=20,
                    label=f"FC>={self.fold_change} & p_val<={self.p_val}")

        plt.scatter(dge_results_df.loc[downregulated_gene, 'log2FoldChange'],
                    dge_results_df.loc[downregulated_gene, '-log10(pvalue)'],
                    color='blue',
                    alpha=0.2,
                    s=20,
                    label=f'FC<=-{self.fold_change} & p_val<={self.p_val}')

        plt.scatter(dge_results_df.loc[unsignificant_gene, 'log2FoldChange'],
                    dge_results_df.loc[unsignificant_gene, '-log10(pvalue)'],
                    color='mediumorchid',
                    alpha=0.2,
                    s=20,
                    label=f'p_val>{self.p_val}')

        plt.scatter(
            dge_results_df.loc[signi_genes_between_up_n_down, 'log2FoldChange'],
            dge_results_df.loc[signi_genes_between_up_n_down, '-log10(pvalue)'],
            color='green',
            alpha=0.2,
            s=20,
            label=
            f'-{self.fold_change}<FC<{self.fold_change} & p_val<={self.p_val}')

        plt.xlabel('Log2 Fold Change', fontweight='bold')
        plt.ylabel('-Log10(p-value)', fontweight='bold')
        plt.title(
            f'Psedobulk_DGE of "{cell_type}" in {self.factor_categories[0]} vs {self.factor_categories[1]}',
            fontweight='bold')
        plt.grid(False)
        plt.axhline(neg_log10_pval,
                    color='black',
                    linestyle='--',
                    label=f'p_val ({self.p_val})')
        plt.axvline(log2_fold_chnage,
                    color='red',
                    linestyle='--',
                    alpha=0.4,
                    label=f'Log2 Fold Change (+{self.fold_change})')
        plt.axvline(-log2_fold_chnage,
                    color='blue',
                    linestyle='--',
                    alpha=0.4,
                    label=f'Log2 Fold Change (-{self.fold_change})')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if self.y_lim_tuple:
            plt.ylim(bottom=self.y_lim_tuple[0], top=self.y_lim_tuple[1])
        _cell_type = cell_type.replace(" ", "")
        factor_0 = (self.factor_categories[0]).replace(" ", "")
        factor_1 = (self.factor_categories[1]).replace(" ", "")
        plt.savefig(path.join(
            dirpath, f'pbkDGE_{_cell_type}_{factor_0}_vs_{factor_1}.svg'),
                    bbox_inches='tight')

    def generate_analysis(self, test_data: Union[AnnData, AnnCollection],
                          dirpath: str, **kwargs):
        '''This method calls methods to perform differential gene expression analysis on data.
        
        Args:
            test_data: AnnData.
            dirpath: Path to save the result.
    
        Returns:
            Pandas DataFrame object containing differential gene expression stats.
        '''

        logger = EventLogger('Differential Gene expression analysis',
                             stdout=self.stdout)
        logger.heading2("DGE analysis using Pseudobulk")
        dirpath = os.path.join(dirpath, 'pseudobulk_dge_result')
        os.makedirs(dirpath, exist_ok=True)
        test_data.obs[self.design_factor] = test_data.obs[
            self.design_factor].str.replace('_', '-')
        assert self.celltype_column in test_data.obs.columns, f"{self.celltype_column} must be a column name in `adata.obs`"
        assert self.design_factor in test_data.obs.columns, f"{self.design_factor} must be a column name in `adata.obs`"
        assert self.sum_column in test_data.obs.columns, f"{self.sum_column} must be a column name in `adata.obs`"
        assert self.factor_categories[0] in test_data.obs[
            self.design_factor].unique(
            ), f"{self.factor_categories[0]} must belong to {self.design_factor}"
        assert self.factor_categories[1] in test_data.obs[
            self.design_factor].unique(
            ), f"{self.factor_categories[1]} must belong to '{self.design_factor}' column"

        cell_type_list = [
            condition
            for condition in test_data.obs[self.celltype_column].unique()
        ]

        if self.cell_subsets:
            cell_type_list = self.cell_subsets

        for cell_type in cell_type_list:
            assert cell_type in test_data.obs[self.celltype_column].unique(
            ), f"{cell_type} must belong to '{self.celltype_column}' column"
            logger.info(f'\nProcessing for "{cell_type}" ...')
            design_matrix = self._make_design_matrix(test_data, cell_type)
            dge_results_df = self.get_differential_expression_results(
                design_matrix, cell_type, dirpath)

            plot = self.plot_volcano(dge_results_df, cell_type, dirpath)
            plt.close(plot)
        logger.info(f"\nPseudobulk-DGE results stored at: {dirpath}\n")

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params for DgePseudoBulk_config."""
        return dict(celltype_column='celltype_column',
                    design_factor='design_factor',
                    factor_categories='factor_categories',
                    sum_column='sum_column',
                    cell_subsets=None,
                    min_cell_threshold=1,
                    fold_change=1.5,
                    p_val=0.05,
                    y_lim_tuple=None,
                    save_plot=True,
                    stdout=False)
