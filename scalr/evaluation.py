import os
from typing import Optional

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import shap
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, RocCurveDisplay, roc_curve, auc
from torch.utils.data import DataLoader
from torch import nn

from .model import LinearModel, CustomShapModel
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
                                            filepath: str,
                                            mapping: Optional[dict] = None):
    """
    Function to generate a classificaton report from the predicted data
    at filepath as classification_report.csv

    Args:
        test_labels: true labels from test set
        pred_labels: predicted labels from trained model
        filepath: path to store classification_report.csv
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
    report.to_csv(f'{filepath}/classification_report.csv')

    return report


def conf_matrix(test_labels: list[int],
                pred_labels: list[int],
                mapping: Optional[dict] = None):
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


def save_top_genes_and_heatmap(
    model: LinearModel,
    test_dl: DataLoader,
    classes: list,
    dirpath: str,
    device: str = 'cpu',
    top_n: int = 50,
) -> None:
    """
    Function to save top n genes of each class and save heatmap of gene to class weight.

    Args:
        model: trained model to extract weights from
        test_dl: test dataloader.
        classes: list of class names.
        dirpath: dir where shap analysis csv & heatmap stored.
        device: device for pytorch.
        top_n: save top n genes based on shap values.
    """
    model.to(device)
    shap_model = CustomShapModel(model)
    explainer = shap.DeepExplainer(shap_model,
                                   next(iter(test_dl))[0].to(device))

    shap_values = []
    for batch in test_dl:
        batch_shap_values = explainer.shap_values(batch[0].to(device))
        shap_values.append(batch_shap_values)

    concate_shap_values = np.concatenate(shap_values).mean(axis=0)
    genes_class_shap_df = pd.DataFrame(concate_shap_values,
                                       index=test_dl.dataset.var_names,
                                       columns=classes)

    class_top_genes = {}
    for class_name in classes:
        sorted_genes = genes_class_shap_df[class_name].sort_values(
            ascending=False)
        class_top_genes[class_name] = sorted_genes.index[:top_n]

    pd.DataFrame(class_top_genes).to_csv(
        os.path.join(dirpath, "shap_analysis.csv"), index=False)

    top_n_genes_heatmap(genes_class_shap_df, dirpath)


def top_n_genes_heatmap(class_genes_weights: pd.DataFrame, dirpath: str):
    """
    Generate a heatmap for top n genes across all classes.

    Args:
        class_genes_weights: genes * classes matrix which contains
                             shap_value/weights of each gene to class.
        dirpath: path to store the heatmap image.
    """

    sns.set(rc={'figure.figsize': (9, 12)})
    sns.heatmap(class_genes_weights, vmin=-1e-2, vmax=1e-2)

    plt.savefig(os.path.join(dirpath, "heatmap.png"))


def roc_auc(test_labels: list[int],
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

    # test labels starts with 0 so we need to add 1 in max.
    for class_label in range(max(test_labels) + 1):

        # fpr: False Positive Rate | tpr: True Positive Rate
        fpr, tpr, _ = roc_curve(test_labels_onehot[:, class_label],
                                np.array(pred_score)[:, class_label])

        roc_auc = auc(fpr, tpr)

        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        display.plot()
        os.makedirs(os.path.join(dirpath, "roc_auc"), exist_ok=True)
        plt.savefig(
            os.path.join(dirpath, 'roc_auc',
                         f'{mapping[class_label].replace(" ", "_")}.png'))
