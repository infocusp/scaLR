from os import path
from typing import Tuple

from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import torch
from torch import nn
from torch.utils.data import DataLoader

from _scalr.utils import EventLogger
from _scalr.utils import write_data


def get_accuracy(test_labels: list[int], pred_labels: list[int]) -> float:
    event_logger = EventLogger('Accuracy')
    accuracy = accuracy_score(test_labels, pred_labels)
    event_logger.info(f'Accuracy: {accuracy}')
    return accuracy


def generate_and_save_classification_report(test_labels: list[int],
                                            pred_labels: list[int],
                                            dirpath: str,
                                            mapping: dict = None) -> DataFrame:
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
    event_logger = EventLogger('ClassReport')

    if mapping:
        test_labels = [mapping[x] for x in test_labels]
        pred_labels = [mapping[x] for x in pred_labels]

    report = DataFrame(
        classification_report(test_labels, pred_labels,
                              output_dict=True)).transpose()
    event_logger.info('\nClassification Report:')
    event_logger.info(report)
    write_data(report, path.join(dirpath, 'classification_report.csv'))

    return report
