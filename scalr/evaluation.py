import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, RocCurveDisplay, roc_curve, auc
from torch.utils.data import DataLoader
from torch import nn

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

    Return:
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

    plt.savefig(f"{dirpath}/heatmap.png")
    return top_n_indices, top_n_genes


def roc_auc(test_labels: list,
            pred_score: list,
            dirpath: str,
            mapping: Optional[dict] = None):
    """ Calculate ROC-AUC and save plot.

    Args:
        test_labels: true labels from the test dataset.
        pred_score: predictions probabities of each sample for all the classes.
    """
    # convert label predictions list to one-hot matrix.
    test_labels_onehot = data_utils.get_one_hot_matrix(np.array(test_labels))

    for class_label in range(max(test_labels)):

        fpr, tpr, _ = roc_curve(test_labels_onehot[:, class_label],
                                np.array(pred_score)[:, class_label])

        roc_auc = auc(fpr, tpr)

        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        display.plot()
        os.makedirs(f"{dirpath}/roc_auc", exist_ok=True)
        plt.savefig(
            f'{dirpath}/roc_auc/{mapping[class_label].replace(" ", "_")}.png'
        )
