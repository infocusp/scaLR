"""This file generates accuracy, classification report and stores it."""

from os import path

from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import torch
from torch import nn
from torch.utils.data import DataLoader

from scalr.utils import EventLogger
from scalr.utils import write_data


def get_accuracy(test_labels: list[int], pred_labels: list[int]) -> float:
    """A function to get accuracy for the predicted labels.

    Args:
        test_labels (list[int]): True labels from the test set.
        pred_labels (list[int]): Predicted labels from the trained model.

    Returns:
        float: accuracy score
    """
    event_logger = EventLogger('Accuracy')
    accuracy = accuracy_score(test_labels, pred_labels)
    event_logger.info(f'Accuracy: {accuracy}')
    return accuracy


def generate_and_save_classification_report(test_labels: list[int],
                                            pred_labels: list[int],
                                            dirpath: str,
                                            mapping: dict = None) -> DataFrame:
    """A function to generate a classificaton report from the actual and predicted data
    and store at `dirpath`.

    Args:
        test_labels: True labels from the test set.
        pred_labels: Predicted labels from the trained model.
        dirpath: Path to store classification_report.
        mapping[optional]: Mapping of label_id to true label_names (id2label).

    Returns:
        A Pandas DataFrame with the classification report.
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
