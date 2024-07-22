from copy import deepcopy
import json
import os
from os import path
from typing import Optional, Union, Tuple

import anndata as ad
from anndata import AnnData
from anndata.experimental import AnnCollection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import scanpy as sc
import seaborn as sns
import shap
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, RocCurveDisplay, roc_curve, auc
import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import LinearModel, CustomShapModel
from .utils import data_utils
from scalr.feature_selection import extract_top_k_features


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


def generate_and_save_classification_report(
        test_labels: list[int],
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

    report = DataFrame(
        classification_report(test_labels, pred_labels,
                              output_dict=True)).transpose()
    print(report)
    report.to_csv(path.join(dirpath, 'classification_report.csv'))

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


def is_early_stop(
    batch_id: int,
    genes_class_shap_df: pd.DataFrame,
    prev_top_genes_batch_wise: dict,
    early_stop_config: dict,
    classes: list,
) -> Tuple[bool, dict]:
    """Function to check whether previous and current batches' common genes are
        are greater greater than or equal to the threshold and return top genes
        batch wise.

    Args:
        batch_id: Current batch number.
        genes_class_shap_df: label/class wise genes shap values(mean across samples).
        prev_top_genes_batch_wise: dict where prev batch's per labels top genes are stored.
        early_stop_config: early stopping config.
        classes: list classes/labels.

    Returns:
        early stop value, top genes batch wise.
    """

    early_stop = True
    top_genes_batch_wise = {}
    for label in classes:
        top_genes_batch_wise[label] = genes_class_shap_df[label].sort_values(ascending=False)[:early_stop_config['top_genes']].index

        # Start checking after first batch.
        if batch_id >= 1:
            num_common_genes = len(
              set(top_genes_batch_wise[label]).intersection(
              set(prev_top_genes_batch_wise[label]))
            )
            # If commnon genes are less than 90 early stop will be false.
            if num_common_genes < early_stop_config['threshold']:
                early_stop = False
        else:
            early_stop = False

    return early_stop, top_genes_batch_wise

def get_top_n_genes(
    model: LinearModel,
    train_dl: DataLoader,
    test_dl: DataLoader,
    classes: list,
    dirpath: str,
    early_stop_config: dict,
    device: str = 'cpu',
    top_n: int = 20,
    n_background_tensor: int = 1000,
) -> None:
    """
    Function to get top n genes of each class and its weights.

    Args:
        model: trained model to extract weights from
        train_dl: train dataloader.
        test_dl: test dataloader.
        classes: list of class names.
        dirpath: dir where genes to class weights stored.
        early_stop_config: early stopping configurations.
        device: device for pytorch.
        top_n: save top n genes based on shap values.
        n_background_tensor: number of background samples for shap.

    Returns:
        class wise top n genes, genes * class weights matrix
    """

    model.to(device)
    shap_model = CustomShapModel(model)

    random_background_data = next(iter(train_dl))[:-1]
    random_background_data = [data.to(device) for data in random_background_data]

    explainer = shap.DeepExplainer(
        shap_model,
        *random_background_data)

    abs_prev_top_genes_batch_wise = {}
    count_patience = 0
    for batch_id, batch in enumerate(test_dl):
        batch_shap_values = explainer.shap_values(batch[0].to(device))

        abs_mean_shap_values = np.abs(batch_shap_values).mean(axis=0)
        # calcluating 2 mean with abs values and non-abs values.
        # Non-abs values required for heatmap.
        mean_shap_values = batch_shap_values.mean(axis=0)

        if batch_id >= 1:
            abs_mean_shap_values = np.mean([abs_mean_shap_values, abs_prev_batches_mean_shap_values], axis=0)
            mean_shap_values = np.mean([mean_shap_values, prev_batches_mean_shap_values], axis=0)

        abs_genes_class_shap_df = DataFrame(abs_mean_shap_values,
                                        index=test_dl.dataset.var_names,
                                        columns=classes)

        abs_prev_batches_mean_shap_values = abs_mean_shap_values
        prev_batches_mean_shap_values = mean_shap_values

        early_stop, abs_prev_top_genes_batch_wise = is_early_stop(
            batch_id, abs_genes_class_shap_df, abs_prev_top_genes_batch_wise,
            early_stop_config, classes
        )

        count_patience = count_patience + 1 if early_stop else 0

        if count_patience == early_stop_config['patience']:
            print(f"Early stopping at batch: {batch_id}")
            break

    genes_class_shap_df = DataFrame(mean_shap_values,
                                    index=test_dl.dataset.var_names,
                                    columns=classes)

    abs_genes_class_shap_df.T.to_csv(
        path.join(dirpath, "genes_class_weights.csv"))

    genes_class_shap_df.T.to_csv(
        path.join(dirpath, "raw_genes_class_weights.csv"))

    # Extract only top N genes
    class_top_genes = {class_label:genes[:top_n]
        for class_label, genes in abs_prev_top_genes_batch_wise.items()
    }

    return class_top_genes, genes_class_shap_df


def save_top_genes_and_heatmap(
    model: LinearModel,
    train_dl: DataLoader,
    test_dl: DataLoader,
    classes: list,
    dirpath: str,
    early_stop_config: dict,
    device: str = 'cpu',
    top_n: int = 20,
    n_background_tensor: int = 1000,
) -> None:
    """
    Function to save top n genes of each class and save heatmap of genes & their class weight.

    Args:
        model: trained model to extract weights from
        train_dl: train dataloader.
        test_dl: test dataloader.
        classes: list of class names.
        dirpath: dir where shap analysis csv & heatmap stored.
        early_stop_config: early stopping configurations.
        device: device for pytorch.
        top_n: save top n genes based on shap values.
        n_background_tensor: number of background samples for shap.
    """

    shap_heatmap_path = path.join(dirpath, "shap_heatmap")
    os.makedirs(shap_heatmap_path, exist_ok=True)

    class_top_genes, genes_class_shap_df = get_top_n_genes(
        model,
        train_dl,
        test_dl,
        classes,
        shap_heatmap_path,
        early_stop_config,
        device,
        top_n,
        n_background_tensor,
    )

    DataFrame(class_top_genes).to_csv(path.join(shap_heatmap_path,
                                                "shap_analysis.csv"),
                                      index=False)

    common_genes = set()
    for class_name, genes in class_top_genes.items():
        common_genes.update(genes)
    plot_heatmap(genes_class_shap_df.loc[list(common_genes)],
                 shap_heatmap_path)


def plot_heatmap(class_genes_weights: DataFrame, dirpath: str):
    """
    Generate a heatmap for top n genes across all classes.

    Args:
        class_genes_weights: genes * classes matrix which contains
                             shap_value/weights of each gene to class.
        dirpath: path to store the heatmap image.
    """

    sns.set(rc={'figure.figsize': (9, 12)})
    sns.heatmap(class_genes_weights, vmin=-1e-2, vmax=1e-2)

    plt.savefig(path.join(dirpath, "heatmap.png"))


def plot_roc_auc_curve(test_labels: list[int],
                       pred_score: list[list[float]],
                       dirpath: str,
                       mapping: Optional[list] = None) -> None:
    """ Calculate ROC-AUC and save plot.

    Args:
        test_labels: true labels from the test dataset.
        pred_score: predictions probabities of each sample for all the classes.
    """
    # convert label predictions list to one-hot matrix.
    test_labels_onehot = data_utils.get_one_hot_matrix(np.array(test_labels))
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # test labels starts with 0 so we need to add 1 in max.
    for class_label in range(max(test_labels) + 1):

        # fpr: False Positive Rate | tpr: True Positive Rate
        fpr, tpr, _ = roc_curve(test_labels_onehot[:, class_label],
                                np.array(pred_score)[:, class_label])

        roc_auc = auc(fpr, tpr)

        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        display.plot(ax=ax, name=mapping[class_label])

    plt.axline((0, 0), (1, 1), linestyle='--', color='black')
    fig.savefig(path.join(dirpath, f'roc_auc.png'))
    plt.clf()  # clear axis & figure so it does not affect the next plot.


def _make_design_matrix(adata: Union[AnnData, AnnCollection],
                        fixed_column: str, fixed_condition: str,
                        design_factor: str, factor_categories: list[str],
                        sum_column: str) -> AnnData:
    """Function to make design matrix based upon fixed and control conditions.

    Args:
        adata: data to make design matrix from
        fixed_column: column name in `adata.obs` containing a fixed condition to subset
        fixed_condition: condition to subset data on, belonging to `fixed_column`
        design_factor: column name in `adata.obs` containing different factor levels or categories
        factor_categories: list of conditions in `design_factor` to make design matrix for
        sum_column: column name to sum values across samples

    Returns:
        AnnData oject of design matrix
    """
    if isinstance(adata, AnnData):
        adata = AnnCollection([adata])

    design_matrix_list = []

    # Hacky fix since deseq2 does not like `_` in column_names
    design_factor = design_factor.replace('_', '')

    fix_data = adata[adata.obs[fixed_column] == fixed_condition]
    for condition in factor_categories:
        condition_subset = fix_data[fix_data.obs[design_factor] == condition]
        for sum_sample in condition_subset.obs[sum_column].unique():
            sum_subset = condition_subset[condition_subset.obs[sum_column] ==
                                          sum_sample]
            subdata = ad.AnnData(
                X=sum_subset[:].X.sum(axis=0).reshape(
                    1, len(sum_subset.var_names)),
                var=DataFrame(index=sum_subset.var_names),
                obs=DataFrame(index=[f'{sum_sample}_{condition}']))
            subdata.obs[design_factor] = [condition]
            design_matrix_list.append(subdata)

    design_matrix = ad.concat(design_matrix_list)
    return design_matrix


def get_differential_expression_results(adata: AnnData,
                                        fixed_condition: str,
                                        design_factor: str,
                                        factor_categories: list[str],
                                        dirpath: str = None) -> DataFrame:
    """Function to get differential expression analysis values

    Args:
        adata: design matrix
        fixed_condition: condition to subset data on, belonging to `fixed_column`
        design_factor: column name in `adata.obs` containing different factor levels or categories
        factor_categories: list of conditions in `design_factor` of design matrix
        dirpath: directory path to store DEG

    Returns:
        pandas DataFrame object containing differential expression stats
    """

    count_matrix = adata.to_df()
    count_matrix_int = count_matrix.round().astype(int)
    metadata_ad = adata.obs

    # Hacky fix since deseq2 does not like `_` in column_names
    design_factor = design_factor.replace('_', '')

    deseq_dataset = DeseqDataSet(counts=count_matrix_int,
                                 metadata=metadata_ad,
                                 design_factors=design_factor)

    # removing genes with zero expression across all the sample
    sc.pp.filter_genes(deseq_dataset, min_cells=1)
    deseq_dataset.deseq2()

    result_stats = DeseqStats(deseq_dataset,
                              contrast=(design_factor, *factor_categories))
    result_stats.summary()
    results_df = result_stats.results_df
    results_df.index.name = 'Gene_id'

    if dirpath:
        results_df.to_csv(
            path.join(
                dirpath,
                f'DEG_results_{fixed_condition}_{factor_categories[0]}_vs_{factor_categories[1]}.csv'
            ))

    return results_df


def plot_volcano(deg_results_df: DataFrame,
                 fixed_condition: str,
                 factor_categories: list[str],
                 fold_change: Union[float, int] = 1.5,
                 p_val: Union[float, int] = 0.05,
                 y_lim_tuple: Optional[Tuple[float, ...]] = None,
                 dirpath: str = None):
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

    upregulated_gene = (deg_results_df['log2FoldChange'] >= log2_fold_chnage
                        ) & (deg_results_df['-log10(pvalue)']
                             >= (neg_log10_pval))
    downregulated_gene = (deg_results_df['log2FoldChange'] <= (
        -log2_fold_chnage)) & (deg_results_df['-log10(pvalue)']
                               >= (neg_log10_pval))

    unsignificant_gene = deg_results_df['-log10(pvalue)'] <= (neg_log10_pval)
    signi_genes_between_up_n_down = ~(upregulated_gene | downregulated_gene
                                      | unsignificant_gene)

    plt.figure(figsize=(10, 6))
    plt.scatter(deg_results_df.loc[upregulated_gene, 'log2FoldChange'],
                deg_results_df.loc[upregulated_gene, '-log10(pvalue)'],
                color='red',
                alpha=0.2,
                s=20,
                label=f"FC>={fold_change} & p_val<={p_val}")

    plt.scatter(deg_results_df.loc[downregulated_gene, 'log2FoldChange'],
                deg_results_df.loc[downregulated_gene, '-log10(pvalue)'],
                color='blue',
                alpha=0.2,
                s=20,
                label=f'FC<=-{fold_change} & p_val<={p_val}')

    plt.scatter(deg_results_df.loc[unsignificant_gene, 'log2FoldChange'],
                deg_results_df.loc[unsignificant_gene, '-log10(pvalue)'],
                color='mediumorchid',
                alpha=0.2,
                s=20,
                label=f'p_val>{p_val}')

    plt.scatter(deg_results_df.loc[signi_genes_between_up_n_down,
                                   'log2FoldChange'],
                deg_results_df.loc[signi_genes_between_up_n_down,
                                   '-log10(pvalue)'],
                color='green',
                alpha=0.2,
                s=20,
                label=f'-{fold_change}<FC<{fold_change} & p_val<={p_val}')

    plt.xlabel('Log2 Fold Change', fontweight='bold')
    plt.ylabel('-Log10(p-value)', fontweight='bold')
    plt.title(
        f'DEG of "{fixed_condition}" in {factor_categories[0]} v/s {factor_categories[1]}',
        fontweight='bold')
    plt.grid(False)
    plt.axhline(neg_log10_pval,
                color='black',
                linestyle='--',
                label=f'p_val ({p_val})')
    plt.axvline(log2_fold_chnage,
                color='red',
                linestyle='--',
                alpha=0.4,
                label=f'Log2 Fold Change (+{fold_change})')
    plt.axvline(-log2_fold_chnage,
                color='blue',
                linestyle='--',
                alpha=0.4,
                label=f'Log2 Fold Change (-{fold_change})')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if y_lim_tuple:
        plt.ylim(bottom=y_lim_tuple[0], top=y_lim_tuple[1])
    plt.savefig(path.join(
        dirpath,
        f'DEG_plot_{fixed_condition}_{factor_categories[0]}_vs_{factor_categories[1]}.png'
    ),
                bbox_inches='tight')


def perform_differential_expression_analysis(adata: Union[AnnData,
                                                          AnnCollection],
                                             fixed_column: str,
                                             fixed_condition: str,
                                             design_factor: str,
                                             factor_categories: list[str],
                                             sum_column: str,
                                             fold_change: Union[float,
                                                                int] = 1.5,
                                             p_val: Union[float, int] = 0.05,
                                             y_lim_tuple: Optional[Tuple[
                                                 float, ...]] = None,
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
    assert fixed_condition in adata.obs[fixed_column].unique(
    ), f"{fixed_condition} must belong to {fixed_column}"
    assert factor_categories[0] in adata.obs[design_factor].unique(
    ), f"{factor_categories[0]} must belong to {design_factor}"
    assert factor_categories[1] in adata.obs[design_factor].unique(
    ), f"{factor_categories[1]} must belong to {design_factor}"

    design_matrix = _make_design_matrix(adata, fixed_column, fixed_condition,
                                        design_factor, factor_categories,
                                        sum_column)

    deg_results_df = get_differential_expression_results(
        design_matrix, fixed_condition, design_factor, factor_categories,
        dirpath)

    plot_volcano(deg_results_df, fixed_condition, factor_categories,
                 fold_change, p_val, y_lim_tuple, dirpath)

    return deg_results_df


def plot_gene_recall(ranked_genes_df: pd.DataFrame,
                     ref_genes_df: pd.DataFrame,
                     top_K: int = 5000,
                     dirpath: str = '.',
                     plot_type: str = '',
                     plots_per_row=5):
    """This function stores the gene recall curve for provided ranked genes & reference genes.
    It also stores the reference genes along with their ranks in a json file for further
    analysis to user.

    Args:
        ranked_genes_df: Pipeline generated ranked genes dataframe.
        ref_genes_df: Reference genes dataframe.
        top_K: The top K ranked genes in which reference genes are to be looked for.
        dirpath: Path to store gene recall plot and json.
        plot_type: Type of gene recall - per category or aggregated across all categories.
    """
    gene_recall_dict = {}

    n_plots = len(
        set(ref_genes_df.columns).intersection(ranked_genes_df.columns))
    print(
        f'-- {n_plots} categories matches between ranked genes & reference genes dataframes, namely: {set(ref_genes_df.columns).intersection(ranked_genes_df.columns)}'
    )
    n_cols = min(plots_per_row, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axs = plt.subplots(n_rows,
                            n_cols,
                            figsize=(n_cols * plots_per_row,
                                     n_rows * plots_per_row),
                            squeeze=False)
    axs = axs.flatten()
    fig.suptitle('Recall of ref genes w.r.t pipeline ranked genes')

    for i, category in enumerate(
            set(ranked_genes_df.columns).intersection(ref_genes_df.columns)):
        ranked_genes = ranked_genes_df[category].values
        ref_genes = ref_genes_df[category].values
        k = top_K

        assert k >= len(
            ref_genes
        ), f'k={k} should be greater than #ref genes({len(ref_genes)})'

        if len(ranked_genes) == 0 or len(ref_genes) == 0:
            raise Exception('Ranked genes or ref genes list cannot be empty.')

        # Adjusting k if expected k > number of ranked genes.
        k = min(k, len(ranked_genes))

        # Building baseline curve
        step = k // len(ref_genes)
        baseline = [i // step for i in range(1, 1 + k)]

        order_in_lit = {}
        points = []

        count = 0
        for rank, gene in enumerate(ranked_genes[:k]):
            if gene in ref_genes:
                count += 1
                order_in_lit[rank] = gene
            points.append(count)

        axs[i].plot(list(range(1, 1 + k)), points, label=f'gene_recall')
        axs[i].plot(list(range(1, 1 + k)), baseline, label='baseline')
        axs[i].set_title(category)
        axs[i].set_xlabel('# Top-ranked genes (k)')
        axs[i].set_ylabel('# Reference genes')
        axs[i].legend()

        gene_recall_dict[category] = order_in_lit

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig(f'{dirpath}/gene_recall_curves_{plot_type}.png')
    with open(f'{dirpath}/gene_recall_curve_{plot_type}_info.json', 'w') as f:
        json.dump(gene_recall_dict, f, indent=6)
    print(
        f'Gene recall curves stored at path : `{dirpath}/gene_recall_curves_{plot_type}.png`'
    )

    plt.close()


def validate_gene_recall_config_and_extract_genes(gene_recall_config, dirpath):

    ranked_genes = {}
    reference_genes = {}
    fcw_path = None
    top_K = gene_recall_config.get('top_K', None)

    if 'reference_genes' not in gene_recall_config:
        raise KeyError(
            'Reference genes information not provided by user in the gene recall config. Please provide or check the README for the same!'
        )
    elif not gene_recall_config['reference_genes']:
        raise ValueError(
            'Atleast one of `per_category` or `aggregated across all categories` reference genes list needs to be provided by user!'
        )
    else:
        if gene_recall_config['reference_genes'].get('per_category', []):
            reference_genes['per_category'] = pd.read_csv(
                gene_recall_config['reference_genes']['per_category'],
                index_col=0)
        else:
            reference_genes['per_category'] = None
        if gene_recall_config['reference_genes'].get('aggregated', []):
            reference_genes['aggr_all_categories'] = pd.read_csv(
                gene_recall_config['reference_genes']['aggregated'],
                index_col=0)
        else:
            reference_genes['aggr_all_categories'] = None

    print('-- Reference genes are extracted.')

    if 'feature_class_weights_path' in gene_recall_config and 'ranked_genes' in gene_recall_config:
        raise ValueError(
            'Either of `feature_class_weights` or `ranked_genes` is expected, not both!'
        )

    if 'feature_class_weights_path' in gene_recall_config:
        fcw_path = gene_recall_config['feature_class_weights_path']
    elif 'feature_class_weights_path' not in gene_recall_config and 'ranked_genes' not in gene_recall_config:
        fcw_path = f'{dirpath}/shap_heatmap/genes_class_weights.csv'

    if fcw_path:
        try:
            feature_class_weights = pd.read_csv(fcw_path, index_col=0)
        except:
            raise KeyError(
                'Ranked genes information not provided by user in the gene recall config. '
                'Nor does the feature class weights matrix is found at expected path '
                f'after the pipeline run - {fcw_path}. Please check the README!'
            )

        if reference_genes['per_category'] is not None:
            ranked_genes['per_category'] = extract_top_k_features(
                feature_class_weights=feature_class_weights,
                aggregation_strategy='no_reduction',
                k=top_K,
                save_features=False)
        else:
            ranked_genes['per_category'] = None
        if reference_genes['aggr_all_categories'] is not None:
            ranked_genes['aggr_all_categories'] = extract_top_k_features(
                feature_class_weights=feature_class_weights,
                aggregation_strategy='mean',
                k=top_K,
                save_features=False).to_frame(
                    name=reference_genes['aggr_all_categories'].columns.
                    to_list()[0])
        else:
            ranked_genes['aggr_all_categories'] = None

    if not fcw_path and 'ranked_genes' in gene_recall_config:
        if gene_recall_config['ranked_genes'].get('per_category', []):
            ranked_genes['per_category'] = pd.read_csv(
                gene_recall_config['ranked_genes']['per_category'],
                index_col=0)
        else:
            if 'per_category' in gene_recall_config['reference_genes']:
                raise KeyError(
                    'Please provide ranked genes list for per_category...')
            ranked_genes['per_category'] = None
        if gene_recall_config['ranked_genes'].get('aggregated', []):
            ranked_genes['aggr_all_categories'] = pd.read_csv(
                gene_recall_config['ranked_genes']['aggregated'], index_col=0)
        else:
            if 'aggregated' in gene_recall_config['reference_genes']:
                raise KeyError(
                    'Please provide ranked genes list for `aggregated across all categories`...'
                )
            ranked_genes['aggr_all_categories'] = None
    print('-- Ranked genes are extracted.')

    return ranked_genes, reference_genes


def generate_gene_recall_curve(gene_recall_config, resultpath):
    """This function geneerates gene recall curves for each category of target provided &
    also plots gene recall for aggregated ranked genes across all categories if user intents to.

    Args:
        gene_recall_config: Config for gene recall.
        resultpath: Path to fetch experiment's stored shap results.
    """

    # Validating the parameters provided in gene recal config and extract ranked & reference gene lists.
    ranked_genes, reference_genes = validate_gene_recall_config_and_extract_genes(
        gene_recall_config, resultpath)

    # Plotting gene recall curves for each category in trait.
    if 'per_category' in gene_recall_config['reference_genes']:
        print('Plotting gene recall curve for each category in the target.')
        plot_gene_recall(ranked_genes['per_category'],
                         reference_genes['per_category'],
                         len(ranked_genes['per_category']),
                         resultpath,
                         plot_type='per_category',
                         plots_per_row=gene_recall_config.get(
                             'plots_per_row', 5))

    # Plotting gene recall curves aggregated across all categories in trait.
    if 'aggregated' in gene_recall_config['reference_genes']:
        print(
            'Plotting gene recall curve for genes ranked by aggregation across all categories in the trait.'
        )
        plot_gene_recall(ranked_genes['aggr_all_categories'],
                         reference_genes['aggr_all_categories'],
                         len(ranked_genes['aggr_all_categories']),
                         resultpath,
                         plot_type='aggregated_across_all_categories',
                         plots_per_row=gene_recall_config.get(
                             'plots_per_row', 5))
