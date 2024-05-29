import os
from os import path
from typing import Optional, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import anndata as ad
from anndata import AnnData
from anndata.experimental import AnnCollection
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, RocCurveDisplay, roc_curve, auc
import torch
from torch import nn
from torch.utils.data import DataLoader
import seaborn as sns
import scanpy as sc

from .model import LinearModel
from .utils import data_utils


def get_predictions(model: LinearModel,
                    test_dl: DataLoader,
                    device: str = 'cpu') -> (list[int], list[int]):
    """
    Function to get classificaton predictions from a model and test_dataloader

    Args:
        model: model to get predictions from
        test_dl: test dataloader containing data as well as true labels
        device: device to run model on ('cpu'/'cuda')

    Returns:
        List of true labels from test set, predicted labels via inference on test
        data and prediction probabilities of each class
    """
    model.eval()
    test_labels, pred_labels, pred_probabilities = [], [], []

    for batch in test_dl:
        with torch.no_grad():
            x, y = [example.to(device)
                    for example in batch[:-1]], batch[-1].to(device)
            out = model(*x)['cls_output']

        test_labels += y.tolist()
        pred_labels += torch.argmax(out, dim=1).tolist()
        pred_probabilities += out.tolist()

    return test_labels, pred_labels, pred_probabilities


# Function to return accuracy score of predicted labels as compared to true labels
# Wrapper for sklearn accuracy_score function
accuracy = accuracy_score


def generate_and_save_classification_report(test_labels: list[int],
                                            pred_labels: list[int],
                                            dirpath: str,
                                            mapping: Optional[dict] = None) -> DataFrame:
    """
    Function to generate a classificaton report from the predicted data
    at dirpath as classification_report.csv

    Args:
        test_labels: true labels from test set
        pred_labels: predicted labels from trained model
        dirpath: path to store classification_report.csv
        mapping[optional]: mapping of label_id to true label_names (id2label)

    Returns:
        a Pandas DataFrame with the classification report
    """

    if mapping is not None:
        test_labels = [mapping[x] for x in test_labels]
        pred_labels = [mapping[x] for x in pred_labels]

    report = pd.DataFrame(
        classification_report(test_labels, pred_labels,
                              output_dict=True)).transpose()
    print(report)
    report.to_csv(path.join(dirpath,'classification_report.csv'))

    return report


def conf_matrix(test_labels: list[int],
                pred_labels: list[int],
                mapping: Optional[dict] = None) -> np.ndarray:
    """
    Function to return confusion matrix of predicted labels as compared to true labels

    Args:
        test_labels: true labels from test set
        pred_labels: predicted labels from trained model
        mapping[optional]: mapping of label_id to true label_names (id2label)

    Returns:
        numpy array of shape (n_classes, n_classes)
    """

    if mapping is not None:
        test_labels = [mapping[x] for x in test_labels]
        pred_labels = [mapping[x] for x in pred_labels]

    return confusion_matrix(test_labels, pred_labels)


# TODO: change extraction method from weights maybe?
def get_top_n_genes(
        model: LinearModel,
        n: int = 50,
        genes: Optional[list[str]] = None) -> (list[int], list[str]):
    """
    Function to get top_n genes and their indices using model weights.

    Args:
        model: trained model to extract weights from
        n: number of top_genes to extract
        genes: gene_name list

    Return:
        top_n gene indices, top_n gene names

    """
    weights = abs(model.state_dict()['lin.weight'].cpu())
    top_n_indices = torch.mean(weights, dim=0).sort().indices[-n:].tolist()
    top_n_genes = []
    if genes is not None: top_n_genes = [genes[i] for i in top_n_indices]
    return top_n_indices, top_n_genes


def top_n_heatmap(model: LinearModel,
                  dirpath: str,
                  classes: list[str],
                  n: int = 50,
                  genes: Optional[list[str]] = None) -> (list[int], list[str]):
    """
    Generate a heatmap for top_n genes across all classes.

    Args:
        model: trained model to extract weights from
        dirpath: path to store the heatmap image
        classes: list of name of classes
        n: number of top_genes to extract
        genes: gene_name list

    Return:
        top_n gene indices, top_n gene names
    """
    weights = model.state_dict()['lin.weight'].cpu()
    top_n_indices, top_n_genes = get_top_n_genes(model, n, genes)
    top_n_weights = weights[:, top_n_indices].transpose(0, 1)

    sns.set(rc={'figure.figsize': (9, 12)})
    sns.heatmap(top_n_weights,
                yticklabels=top_n_genes,
                xticklabels=classes,
                vmin=-1e-2,
                vmax=1e-2)

    plt.savefig(path.join(dirpath,'heatmap.png'))
    return top_n_indices, top_n_genes


def roc_auc(test_labels: list[int],
            pred_score: list[list[float]],
            dirpath: str,
            mapping: Optional[dict] = None) -> None:
    """ Calculate ROC-AUC and save plot.

    Args:
        test_labels: true labels from the test dataset.
        pred_score: predictions probabities of each sample for all the classes.
    """
    # convert label predictions list to one-hot matrix.
    test_labels_onehot = data_utils.get_one_hot_matrix(np.array(test_labels))

    for class_label in range(max(test_labels)):

        # fpr: False Positive Rate | tpr: True Positive Rate
        fpr, tpr, _ = roc_curve(test_labels_onehot[:, class_label],
                                np.array(pred_score)[:, class_label])

        roc_auc = auc(fpr, tpr)

        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        display.plot()
        os.makedirs(os.path.join(dirpath,"roc_auc"), exist_ok=True)
        plt.savefig(
            os.path.join(dirpath,'roc_auc',f'{mapping[class_label].replace(" ", "_")}.png')
        )


def _make_design_matrix(adata: Union[AnnData, AnnCollection],
                        fixed_column: str, fixed_condition: str, design_factor: str,
                        factor_categories: list[str], sum_column: str) -> AnnData:
    """Function to make design matrix based upon fixed and control conditions.

    Args:
        adata: data to make design matrix from
        fixed_column: column name in `adata.obs` containing a fixed condition to subset
        fixed_condition: condition to subset data on, belonging to `fixed_column`
        design_factor: column name in `adata.obs` containing the control condition
        factor_categories: list of conditions in `design_factor` to make design matrix for
        sum_column: column name to sum values across samples

    Returns:
        AnnData oject of design matrix
    """
    if isinstance(adata, AnnData):
        adata = AnnCollection([adata])

    design_matrix_list = []

    # Hacky fix since deseq2 does not like `_` in column_names
    new_control_column_name = design_factor.replace('_','')

    fix_data = adata[adata.obs[fixed_column]==fixed_condition]
    for condition in factor_categories:
        condition_subset = fix_data[fix_data.obs[design_factor]==condition]
        for sum_sample in condition_subset.obs[sum_column].unique():
            sum_subset = condition_subset[condition_subset.obs[sum_column]==sum_sample]
            subdata = ad.AnnData(X = sum_subset[:].X.sum(axis = 0).reshape(1,len(sum_subset.var_names)),
                                 var = pd.DataFrame(index=sum_subset.var_names),
                                 obs = pd.DataFrame(index=[f'{sum_sample}_{condition}']))
            subdata.obs[new_control_column_name] = [condition]
            design_matrix_list.append(subdata)

    design_matrix = ad.concat(design_matrix_list)
    return design_matrix

def get_differential_expression_results(adata: AnnData, fixed_condition: str,
                                        design_factor: str,
                                        factor_categories: list[str],
                                        dirpath:str = None) -> DataFrame:
    """Function to get differential expression analysis values

    Args:
        adata: design matrix
        fixed_condition: condition to subset data on, belonging to `fixed_column`
        design_factor: column name in `adata.obs` containing the control condition
        factor_categories: list of conditions in `design_factor` of design matrix
        dirpath: directory path to store DEG

    Returns:
        pandas DataFrame object containing differential expression stats
    """

    count_matrix = adata.to_df()
    count_matrix_int = count_matrix.round().astype(int)
    metadata_ad = adata.obs

    # Hacky fix since deseq2 does not like `_` in column_names
    design_factor = design_factor.replace('_','')

    deseq_dataset = DeseqDataSet(counts = count_matrix_int,
                                 metadata = metadata_ad,
                                 design_factors = design_factor)


    # removing genes with zero expression across all the sample
    sc.pp.filter_genes(deseq_dataset, min_cells = 1)
    deseq_dataset.deseq2()

    result_stats = DeseqStats(deseq_dataset, contrast=(design_factor, *factor_categories))
    result_stats.summary()
    results_df = result_stats.results_df

    if dirpath:
        results_df.to_csv(f'{dirpath}/DEG_results_{fixed_condition}_{factor_categories[0]}_vs_{factor_categories[1]}.csv')

    return results_df

def plot_volcano(deg_results_df: DataFrame,
                  fixed_condition: str,
                  factor_categories: list[str],
                  fold_change : Union[float,int] = 1.5,
                  p_val : Union[float,int] = 0.05,
                  y_lim_tuple : Optional[Tuple[float, ...]] = None,
                  dirpath:str = None):

    """Function to generate volcano plot differential expression results and store it to disk

    Args:
        deg_results_df: differential expression results dataframe
        fixed_condition: condition to subset data on, belonging to `fixed_column`
        factor_categories: list of conditions in `design_factor` to make design matrix for
        fold_change: fold change to filter the differentially expressed genes for volcano plot
        p_val: p_val to filter the differentially expressed genes for volcano plot
        y_lim_tuple: values to adjust the Y-axis limits of the plot
        dirpath: directory path to store volcano plot
    """
    log2_fold_chnage = np.log2(fold_change)
    neg_log10_pval = -np.log10(p_val)
    deg_results_df['-log10(pvalue)'] = -np.log10(deg_results_df['pvalue'])

    upregulated_gene = (deg_results_df['log2FoldChange']>=log2_fold_chnage)&(deg_results_df['-log10(pvalue)']>=(neg_log10_pval))
    downregulated_gene = (deg_results_df['log2FoldChange']<=(-log2_fold_chnage))&(deg_results_df['-log10(pvalue)']>=(neg_log10_pval))

    unsignificant_gene = deg_results_df['-log10(pvalue)']<=(neg_log10_pval)
    signi_genes_between_up_n_down = ~(upregulated_gene|downregulated_gene|unsignificant_gene)

    plt.figure(figsize=(10, 6))
    plt.scatter(deg_results_df.loc[upregulated_gene,'log2FoldChange'], deg_results_df.loc[upregulated_gene,'-log10(pvalue)'],
                color='red',alpha=0.2,s=20,label= f"FC>={fold_change} & p_val<={p_val}")

    plt.scatter(deg_results_df.loc[downregulated_gene,'log2FoldChange'], deg_results_df.loc[downregulated_gene,'-log10(pvalue)'],
                color='blue',alpha=0.2,s=20,label=f'FC<=-{fold_change} & p_val<={p_val}')

    plt.scatter(deg_results_df.loc[unsignificant_gene,'log2FoldChange'], deg_results_df.loc[unsignificant_gene,'-log10(pvalue)'],
                color='mediumorchid',alpha=0.2,s=20,label=f'p_val>{p_val}')

    plt.scatter(deg_results_df.loc[signi_genes_between_up_n_down,'log2FoldChange'], deg_results_df.loc[signi_genes_between_up_n_down,'-log10(pvalue)'],
                color='green',alpha=0.2,s=20,label=f'-{fold_change}<FC<{fold_change} & p_val<={p_val}')

    plt.xlabel('Log2 Fold Change',fontweight='bold')
    plt.ylabel('-Log10(p-value)',fontweight='bold')
    plt.title(f'DEG of "{fixed_condition}" in {factor_categories[0]} v/s {factor_categories[1]}',fontweight='bold')
    plt.grid(False)
    plt.axhline(neg_log10_pval,color='black', linestyle='--',label=f'p_val ({p_val})')
    plt.axvline(log2_fold_chnage, color='red', linestyle='--', alpha=0.4,label=f'Log2 Fold Change (+{fold_change})')
    plt.axvline(-log2_fold_chnage, color='blue', linestyle='--', alpha=0.4,label=f'Log2 Fold Change (-{fold_change})')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if y_lim_tuple:
        plt.ylim(bottom=y_lim_tuple[0],top=y_lim_tuple[1])
    if dirpath:
        plt.savefig(f'{dirpath}/DEG_plot_{fixed_condition}_{factor_categories[0]}_vs_{factor_categories[1]}.png',
                    bbox_inches='tight')
    else:
        plt.show()

def perform_differential_expression_analysis(adata: Union[AnnData, AnnCollection],
                                      fixed_column: str, fixed_condition: str,
                                      design_factor: str,factor_categories: list[str],
                                      sum_column: str,fold_change: Union[float,int] = 1.5,
                                      p_val: Union[float,int] = 0.05,
                                      y_lim_tuple: Optional[Tuple[float, ...]] = None,
                                      dirpath: str = None) -> DataFrame:
    """Function to perform differential expression analysis on data

    Args:
        adata: data to make design matrix from
        fixed_column: column name in `adata.obs` containing a fixed condition to subset 
        fixed_condition: condition to subset data on, belonging to `fixed_column`
        design_factor: column name in `adata.obs` containing the control condition
        factor_categories: list of conditions in `design_factor` to make design matrix for
        sum_column: column name to sum values across samples
        fold_change: fold change to filter the differentially expressed genes for volcano plot
        p_val: p_val to filter the differentially expressed genes for volcano plot
        y_lim_tuple: values to adjust the Y-axis limits of the plot
        dirpath: directory path to store analysis results and volcano plot

    Returns:
        pandas DataFrame object containing differential expression stats
    """

    assert fixed_column in adata.obs.columns, f"{fixed_column} must be a column name in `adata.obs`"
    assert design_factor in adata.obs.columns, f"{design_factor} must be a column name in `adata.obs`"
    assert sum_column in adata.obs.columns, f"{sum_column} must be a column name in `adata.obs`"
    assert fixed_condition in adata.obs[fixed_column].unique(),f"{fixed_condition} must belong to {fixed_column}"
    assert factor_categories[0] in adata.obs[design_factor].unique(), f"{factor_categories[0]} must belong to {design_factor}"
    assert factor_categories[1] in adata.obs[design_factor].unique(), f"{factor_categories[1]} must belong to {design_factor}"

    design_matrix = _make_design_matrix(adata,fixed_column, fixed_condition,
                  design_factor,factor_categories,sum_column)

    deg_results_df = get_differential_expression_results(design_matrix, fixed_condition,
                                                         design_factor, factor_categories,
                                                         dirpath)

    plot_volcano(deg_results_df, fixed_condition, factor_categories, fold_change,
                  p_val, y_lim_tuple, dirpath)

    return deg_results_df
