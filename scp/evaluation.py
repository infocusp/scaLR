import os
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def predictions(model, test_dl, device='cpu'):
    """
    Function to get classifcaiton predictions from a model and test_dataloader

    Args:
        model: model to get predictions from
        test_dl: test dataloader containing data as well as true labels
        device: device to run model on ('cpu'/'cuda')

    Return:
        true_labels from test set, predicted labels via inference on test data
    """
    model.eval()
    test_labels, pred_labels = [], []

    for batch in test_dl:
        with torch.no_grad():
            x, y = [x_.to(device) for x_ in batch[:-1]], batch[-1].to(device)
            out = model(*x)['cls_output']
    
        test_labels += y.tolist()
        pred_labels += torch.argmax(out,dim=1).tolist()

    return test_labels, pred_labels

def accuracy(test_labels, pred_labels):
    """
    Function to return accuracy score of predicted labels as compared to true labels
    """
    return accuracy_score(test_labels, pred_labels)

def report(test_labels, pred_labels, filepath, mapping=None):
    """
    Function to generate a classifcaiton report from a from the predicted data
    at filepath as classification_report.csv

    Args:
        test_labels: true labels from test set
        pred_labels: predicted labels from trained model
        filepath: path to store classification_report.csv
        mapping[optional]: mapping of label_id to true label_names (id2label)
    """
    
    if mapping is not None:
        test_labels = list(map(lambda x: mapping[x], test_labels))
        pred_labels = list(map(lambda x: mapping[x], pred_labels))
        
    report = pd.DataFrame(classification_report(test_labels, pred_labels, output_dict=True)).transpose()
    print(report)
    report.to_csv(f'{filepath}/classification_report.csv')
    return

def conf_matrix(test_labels, pred_labels):
    """
    Function to return confusion matrix of predicted labels as compared to true labels
    """    
    return confusion_matrix(test_labels, pred_labels)

# TODO: change extraction method from weights maybe?
def get_top_n_genes(model, n=50, genes=None):
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

def top_n_heatmap(model, filepath, classes, n=50, genes=None):
    """
    Generate a heatmap for top_n genes across all classes.
    """
    weights = model.state_dict()['lin.weight'].cpu()
    top_n_indices, top_n_genes = get_top_n_genes(model, n, genes)
    top_n_weights = weights[:,top_n_indices].transpose(0, 1)

    sns.set (rc = {'figure.figsize':(9, 12)})
    sns.heatmap(top_n_weights, yticklabels=top_n_genes, xticklabels=classes, vmin=-1e-2, vmax=1e-2)
    
    plt.savefig(f"{filepath}/heatmap.png")
    return top_n_weights, top_n_genes











