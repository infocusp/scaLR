"""This file generates differential gene expression analysis using Linear mixed-effects model approach and stores the results."""
import multiprocessing
import os
from os import path
import pickle
import resource
import string
import traceback
from typing import Optional, Tuple, Union
import warnings

from anndata import AnnData
from anndata import ImplicitModificationWarning
import anndata as ad
from anndata.experimental import AnnCollection
from joblib import delayed
from joblib import Parallel
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import LinAlgError
from pandas import DataFrame
import pandas as pd
import scanpy as sc
from scipy.optimize import OptimizeWarning
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tools.sm_exceptions import HessianInversionWarning

from scalr import utils
from scalr.analysis import AnalysisBase
from scalr.utils import EventLogger
from scalr.utils import read_data


class DgeLMEM(AnalysisBase):
    '''Class to perform differential gene expression analysis 
    using Linear mixed effects model.'''

    def __init__(self,
                 fixed_effect_column: str,
                 fixed_effect_factors: list[str],
                 group: str,
                 celltype_column: str = None,
                 cell_subsets: list[str] = None,
                 min_cell_threshold: int = 10,
                 n_cpu: int = 6,
                 gene_batch_size: int = 1000,
                 coef_threshold: [float, int] = 0,
                 p_val: Union[float, int] = 0.05,
                 y_lim_tuple: Optional[Tuple[float, ...]] = None,
                 save_plot: bool = True,
                 stdout: bool = False):
        '''DgeLMEM parameters initialization.
        
        Args: 
            fixed_effect_column: Column name in `anndata.obs` containing different factor levels or categories for
                           differential gene expression analysis. This acts as a fixed_effect parameter.           
            fixed_effect_factors: List of conditions in `fixed_effect_column` to make design matrix for.
            group: Column name to act as a random_effect parameter for mixed effect model.
            celltype_column: Column name in `anndata.obs` containing all the cell types.
            cell_subsets: Selcted list of cell types in 'celltype_column' to subset the anndata.
            min_cell_threshold: Minimum number of cells with nonzero values for a gene. Used for filtering noisy genes.
            n_cpu: Number of CPUs for parallelization.
            gene_batch_size: Number of genes in a batch of process.
            coef_threshold: Threshold to filter up and down-regulated genes in volcano plot.
            p_val: p value, to filter the differentially expressed genes for the volcano plot.
            y_lim_tuple: Values to adjust the Y-axis limits of the plot.
            save_plot: Boolean value to save plot.
            stdout : Flag to print logs to stdout.
        '''

        self.fixed_effect_column = fixed_effect_column
        self.fixed_effect_factors = fixed_effect_factors[::-1]
        self.group = group
        self.celltype_column = celltype_column
        self.cell_subsets = cell_subsets
        self.min_cell_threshold = min_cell_threshold
        self.n_cpu = n_cpu
        self.gene_batch_size = gene_batch_size
        self.coef_threshold = coef_threshold
        self.p_val = p_val
        self.y_lim_tuple = y_lim_tuple
        self.save_plot = save_plot
        self.stdout = stdout

        if self.n_cpu > multiprocessing.cpu_count():
            self.n_cpu = multiprocessing.cpu_count()

    def replace_spec_char_get_dict(self, var_names: pd.core.indexes.base.Index):
        ''' This method replaces any special character in gene names.

        Args:
            var_names: var_names in the Anndata.
            
        Returns:
            var_names with special characters replaced with '_', and a dictionary mapping of old and new names.
        '''

        old_new_name_map_dict = dict()
        special_chars = list(set(string.punctuation))

        def replace_special_chars(name):
            return ''.join(
                '_' if char in special_chars else char for char in name)

        replaced_name_array = np.vectorize(replace_special_chars)(var_names)
        for name in range(len(var_names)):
            old_new_name_map_dict[replaced_name_array[name]] = var_names[name]
        return replaced_name_array, old_new_name_map_dict

    def get_genes_n_fixed_val_subset_df(self,
                                        batch_adata: AnnData,
                                        cell_type: str = None):
        '''This method converts Anndata into a pandas DataFrame with gene expression data,
        'fixed_effect_column', and 'group' params.
        
        Args:
            batch_adata: Anndata.
            cell_type: Cell type in the 'celltype_column' to subset the anndata, 
                       the whole anndata will be processed if 'cell_type' is None.
            
        Returns:
            A list of gene names in the anndata, and a pandas dataframe with count matrix.
        '''

        if cell_type is not None:
            mask = pd.Series([True] * batch_adata.shape[0],
                             index=batch_adata.obs_names)
            for factor in self.fixed_effect_factors:
                assert factor in batch_adata.obs[
                    self.
                    fixed_effect_column].values, f"{factor} must be in the '{self.fixed_effect_column}' column in 'adata.obs'"
            mask &= batch_adata.obs[self.fixed_effect_column].isin(
                self.fixed_effect_factors)
            ad_subset = batch_adata[mask]
            #Remove unexpressed genes
            sc.pp.filter_genes(ad_subset, min_cells=self.min_cell_threshold)
            ad_subset_to_df = ad_subset.to_df()
            genes = ad_subset_to_df.columns.tolist()
            ad_subset_to_df[self.group] = ad_subset.obs[self.group].values
            ad_subset_to_df[self.fixed_effect_column] = ad_subset.obs[
                self.fixed_effect_column].values
        else:
            sc.pp.filter_genes(batch_adata, min_cells=self.min_cell_threshold)
            ad_subset_to_df = batch_adata.to_df()
            genes = ad_subset_to_df.columns.tolist()
            ad_subset_to_df[self.group] = batch_adata.obs[self.group].values
            ad_subset_to_df[self.fixed_effect_column] = batch_adata.obs[
                self.fixed_effect_column].values
        ad_subset_to_df[self.fixed_effect_column] = pd.Categorical(
            ad_subset_to_df[self.fixed_effect_column],
            categories=[*self.fixed_effect_factors],
            ordered=True)

        return genes, ad_subset_to_df

    def get_result_mxmodel_per_gene(self, gene: str,
                                    ad_subset_to_df: DataFrame):
        '''This method produces the Linear mixed-effects model statistics for a single gene.
        
        Args:
            gene: Gene name.
            ad_subset_to_df: A pandas dataframe with gene expression, 'fixed_effect_column', 
                             and 'group' params.
            
        Returns:
            A dictionary with model statistics.
        '''

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", OptimizeWarning)
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", HessianInversionWarning)
                formula = f'{gene} ~ {self.fixed_effect_column}'
                mxmodel = smf.mixedlm(formula,
                                      ad_subset_to_df,
                                      groups=ad_subset_to_df[self.group])
                mixmodel_result = mxmodel.fit()
                result_dict_per_gene = dict()
                result_dict_per_gene['gene'] = gene
                for category in self.fixed_effect_factors[1:]:
                    result_dict_per_gene[
                        f'coef_{category}'] = mixmodel_result.params[
                            f'{self.fixed_effect_column}[T.{category}]']
                    result_dict_per_gene[
                        f'SEcoef_{category}'] = mixmodel_result.bse[
                            f'{self.fixed_effect_column}[T.{category}]']
                    result_dict_per_gene[
                        f'pval_{category}'] = mixmodel_result.pvalues[
                            f'{self.fixed_effect_column}[T.{category}]']
                pickle.dumps(result_dict_per_gene)
                return result_dict_per_gene
        except LinAlgError as e:
            pass

    def get_multiproc_mxeffect_model_batch_res(self, gene_names: list[str],
                                               ad_subset_to_df: DataFrame):
        '''This method parallelizes the Linear mixed effects models for a list
        of genes.
        
        Args:
            gene_names: List of gene names.
            ad_subset_to_df: A pandas dataframe with gene expression, 'fixed_effect_column', 
            and 'group' params.
            
        Returns:
            A list of dictionaries with model stats for 'gene_names'.    
        '''

        mxmodel_results_list = Parallel(n_jobs=self.n_cpu, backend="loky")(
            delayed(self.get_result_mxmodel_per_gene)(gene, ad_subset_to_df)
            for gene in gene_names)
        mxmodel_results_list = [
            per_gene_result for per_gene_result in mxmodel_results_list
            if per_gene_result is not None
        ]
        return mxmodel_results_list

    def plot_lmem_dge_result(self,
                             lmem_res_df: DataFrame,
                             dirpath: str,
                             cell_type: str = None):
        '''This method produces a volcano plot for the model results for a data subset 
        with a cell type, or for the whole dataset.
        
        Args:
            lmem_res_df: A pandas DataFrame with Model results (p-value, co-efficients, Standard error..).
            dirpath: Path to save the plot.
            cell_type: Cell type used to subset input anndata.
        '''

        neg_log10_pval = -np.log10(self.p_val)
        for category in self.fixed_effect_factors:
            coef_col = next((col for col in lmem_res_df.columns
                             if col.startswith('coef') and category in col),
                            None)
            pval_col = next((col for col in lmem_res_df.columns
                             if col.startswith('pval') and category in col),
                            None)
            if (coef_col is not None) and (pval_col is not None):

                lmem_res_df[f'-log10_{pval_col}'] = -np.log10(
                    lmem_res_df[pval_col])
                down_reg_genes_idx = (lmem_res_df[coef_col] < (
                    self.coef_threshold)) & (lmem_res_df[f'-log10_{pval_col}']
                                             >= (neg_log10_pval))
                up_reg_genes_idx = (lmem_res_df[coef_col] > (
                    self.coef_threshold)) & (lmem_res_df[f'-log10_{pval_col}']
                                             >= (neg_log10_pval))
                rest_gene_idx = ~(down_reg_genes_idx | up_reg_genes_idx)
                plt.figure(figsize=(10, 5))
                plt.grid(False)
                plt.scatter(lmem_res_df.loc[up_reg_genes_idx, coef_col],
                            lmem_res_df.loc[up_reg_genes_idx,
                                            f'-log10_{pval_col}'],
                            color='red',
                            s=20,
                            alpha=0.25,
                            label=f"Upreg_genes(p<={self.p_val})")
                plt.scatter(lmem_res_df.loc[down_reg_genes_idx, coef_col],
                            lmem_res_df.loc[down_reg_genes_idx,
                                            f'-log10_{pval_col}'],
                            color='blue',
                            s=20,
                            alpha=0.25,
                            label=f"Downreg_genes(p<={self.p_val})")

                plt.scatter(lmem_res_df.loc[rest_gene_idx, coef_col],
                            lmem_res_df.loc[rest_gene_idx,
                                            f'-log10_{pval_col}'],
                            color='green',
                            alpha=0.2,
                            s=20,
                            label=f"Insignificant genes")
                if self.coef_threshold == 0:
                    plt.axvline(self.coef_threshold,
                                color='red',
                                linestyle='--',
                                alpha=0.4,
                                label=f'Coefficient({self.coef_threshold})')
                    plt.axhline(neg_log10_pval,
                                color='black',
                                linestyle='--',
                                alpha=0.4,
                                label=f'p_val ({self.p_val})')
                if self.coef_threshold != 0:
                    plt.axvline(self.coef_threshold,
                                color='red',
                                linestyle='--',
                                alpha=0.4,
                                label=f'Coefficient({self.coef_threshold})')
                    plt.axvline(-self.coef_threshold,
                                color='blue',
                                linestyle='--',
                                alpha=0.4,
                                label=f'Coefficient(-{self.coef_threshold})')
                    plt.axhline(neg_log10_pval,
                                color='black',
                                linestyle='--',
                                alpha=0.4,
                                label=f'p_val ({self.p_val})')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.6))

                plt.xlabel('Coefficient', fontweight='bold')
                plt.ylabel('-Log10(p-value)', fontweight='bold')
                _category = category.replace(' ', '')
                if cell_type is not None:
                    _cell_type = cell_type.replace(' ', '')
                    plt.title(
                        f'lmem_DGE of "{cell_type}" in "{category}" vs "{self.fixed_effect_factors[0]}"',
                        fontweight='bold')
                    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    plt.savefig(path.join(
                        dirpath, f'lmem_DGE_{_cell_type}_{_category}.svg'),
                                bbox_inches='tight')
                else:
                    plt.title(
                        f'lmem_DGE in "{category}" vs "{self.fixed_effect_factors[0]}"',
                        fontweight='bold')
                    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    plt.savefig(path.join(dirpath, f'lmemDGE_{_category}.svg'),
                                bbox_inches='tight')

    def generate_analysis(self, test_data: Union[AnnData, AnnCollection],
                          dirpath: str, **kwargs):
        '''This method calls methods to run multiple linear mixed effects models and  
        generate DGE results and plots.
        
        Args:
            test_data: Anndata.
            dirpath: Path to save results.
        '''

        if isinstance(test_data, AnnData):
            test_data = AnnCollection([test_data])
        logger = EventLogger('Differential Gene expression analysis',
                             stdout=self.stdout)
        logger.heading2("DGE analysis using LMEM")
        dirpath = os.path.join(dirpath, 'lmem_dge_result')
        os.makedirs(dirpath, exist_ok=True)
        new_var_names, varname_map_dict = self.replace_spec_char_get_dict(
            test_data.var_names)
        test_data.var_names = new_var_names
        result = False
        if self.celltype_column is not None:
            logger.info("Performing DGE analysis with subset anndata")
            fixed_val_list = list(test_data.obs[self.celltype_column].unique())
            if self.cell_subsets is not None:
                fixed_val_list = self.cell_subsets
            for cell_type in fixed_val_list:
                assert cell_type in test_data.obs[
                    self.
                    celltype_column].values, f"{cell_type} must be in the '{self.celltype_column}' column in 'adata.obs'"
                cell_type_test_data = test_data[test_data.obs[
                    self.celltype_column] == cell_type]
                logger.info(f'\nProcessing for "{cell_type}" ...')
                fixed_val_lmem_result_list = []
                for batch in range(0, len(cell_type_test_data.var_names),
                                   self.gene_batch_size):
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", category=ImplicitModificationWarning)
                        gene_list, batch_df = self.get_genes_n_fixed_val_subset_df(
                            cell_type_test_data[:, batch:batch +
                                                self.gene_batch_size].to_adata(
                                                ), cell_type)
                    result_lmem_batch = self.get_multiproc_mxeffect_model_batch_res(
                        gene_list, batch_df)
                    fixed_val_lmem_result_list.extend(result_lmem_batch)

                if fixed_val_lmem_result_list:
                    fixed_val_lmem_result_df = pd.DataFrame(
                        fixed_val_lmem_result_list)
                    standard_error_cols = [
                        col for col in fixed_val_lmem_result_df.columns
                        if col.startswith('SEcoef')
                    ]
                    pval_columns = [
                        col for col in fixed_val_lmem_result_df.columns
                        if col.startswith('pval')
                    ]
                    fixed_val_lmem_result_df.dropna(subset=pval_columns,
                                                    inplace=True)
                    for column in standard_error_cols:
                        condition_category = '_'.join(column.split('_')[1:])
                        fixed_val_lmem_result_df[
                            f'stat_{condition_category}'] = (
                                fixed_val_lmem_result_df[
                                    f'coef_{condition_category}'] /
                                (fixed_val_lmem_result_df[
                                    f'SEcoef_{condition_category}'].replace(
                                        0, np.nan)))
                    for column in pval_columns:
                        multitest_result_bh = multipletests(
                            fixed_val_lmem_result_df[column],
                            method='fdr_bh',
                            alpha=0.1)
                        fixed_val_lmem_result_df[
                            f'adj_{column}'] = multitest_result_bh[1]

                    for gene_name in fixed_val_lmem_result_df['gene']:
                        fixed_val_lmem_result_df[
                            'gene'] = fixed_val_lmem_result_df['gene'].replace(
                                {gene_name: varname_map_dict[gene_name]})
                    file_name = 'lmemDGE_' + (cell_type.replace(" ",
                                                                '')) + '.csv'
                    full_file_path = path.join(dirpath, file_name)
                    fixed_val_lmem_result_df.to_csv(full_file_path, index=False)
                    if self.save_plot:
                        plot = self.plot_lmem_dge_result(
                            fixed_val_lmem_result_df, dirpath, cell_type)
                        plt.close('all')
                    result = True
                else:
                    logger.info(f'Error in producing result for "{cell_type}"')
        else:
            logger.info("Performing DGE analysis with whole anndata ...")
            whole_data_lmem_result_list = []
            for batch in range(0, len(test_data.var_names),
                               self.gene_batch_size):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", category=ImplicitModificationWarning)
                    gene_list, batch_df = self.get_genes_n_fixed_val_subset_df(
                        test_data[:, batch:batch +
                                  self.gene_batch_size].to_adata())
                result_lmem_batch = self.get_multiproc_mxeffect_model_batch_res(
                    gene_list, batch_df)
                whole_data_lmem_result_list.extend(result_lmem_batch)

            if whole_data_lmem_result_list:
                whole_data_lmem_result_df = pd.DataFrame(
                    whole_data_lmem_result_list)
                standard_error_cols = [
                    col for col in whole_data_lmem_result_df.columns
                    if col.startswith('SEcoef')
                ]
                pval_columns = [
                    col for col in whole_data_lmem_result_df.columns
                    if col.startswith('pval')
                ]
                whole_data_lmem_result_df.dropna(subset=pval_columns,
                                                 inplace=True)
                for column in standard_error_cols:
                    condition_category = '_'.join(column.split('_')[1:])
                    whole_data_lmem_result_df[f'stat_{condition_category}'] = (
                        whole_data_lmem_result_df[f'coef_{condition_category}']
                        / (whole_data_lmem_result_df[
                            f'SEcoef_{condition_category}'].replace(0, np.nan)))
                for column in pval_columns:
                    multitest_result_bh = multipletests(
                        whole_data_lmem_result_df[column],
                        method='fdr_bh',
                        alpha=0.1)
                    whole_data_lmem_result_df[
                        f'adj_{column}'] = multitest_result_bh[1]

                for gene_name in whole_data_lmem_result_df['gene']:
                    whole_data_lmem_result_df[
                        'gene'] = whole_data_lmem_result_df['gene'].replace(
                            {gene_name: varname_map_dict[gene_name]})
                file_name = 'lmemDGE_whole' + '.csv'
                full_file_path = path.join(dirpath, file_name)
                whole_data_lmem_result_df.to_csv(full_file_path, index=False)
                if self.save_plot:
                    plot = self.plot_lmem_dge_result(whole_data_lmem_result_df,
                                                     dirpath)
                    plt.close('all')
                result = True
        if result:
            logger.info(f"\nLMEM-DGE results stored at: {dirpath}\n")

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params for DgeLMEM_config."""
        return dict(fixed_effect_column='fixed_effect_column',
                    fixed_effect_factors='fixed_effect_factors',
                    group='group',
                    celltype_column=None,
                    cell_subsets=None,
                    min_cell_threshold=10,
                    n_cpu=6,
                    gene_batch_size=1000,
                    coef_threshold=0,
                    p_val=0.05,
                    y_lim_tuple=None,
                    save_plot=True,
                    stdout=False)
