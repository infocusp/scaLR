"""This file generates ROC-AUC plot and stores it."""

from os import path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay

from scalr.analysis import AnalysisBase
from scalr.utils import data_utils
from scalr.utils import EventLogger


class RocAucCurve(AnalysisBase):
    '''Class to generate ROC-AUC curve.'''

    def generate_analysis(self, test_labels: list[int],
                          pred_probabilities: list[list[float]], dirpath: str,
                          mapping: list, **kwargs) -> None:
        """A function to calculate ROC-AUC and save the plot.

        Args:
            test_labels: True labels from the test dataset.
            pred_probabilities: Predictions probabilities of each sample for all the classes.
            dirpath: Path to store gene recall curve if applicable.
            mapping: List of class names.
        """

        logger_name = "ROC-AUC Analysis"
        self.event_logger = EventLogger(logger_name)
        self.event_logger.heading2(logger_name)
        self.event_logger.info("Generating one hot matrix of test labels.")

        # convert label predictions list to the one-hot matrix.
        test_labels_onehot = data_utils.get_one_hot_matrix(
            np.array(test_labels))
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        self.event_logger.info(
            "Calculating ROC-AUC for each label and creating a plot for that.")

        # test labels start with 0 so we need to add 1 in max.
        for class_label in range(max(test_labels) + 1):

            # fpr: False Positive Rate | tpr: True Positive Rate
            fpr, tpr, _ = roc_curve(
                test_labels_onehot[:, class_label],
                np.array(pred_probabilities)[:, class_label])

            roc_auc = auc(fpr, tpr)

            display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
            display.plot(ax=ax, name=mapping[class_label])

        self.event_logger.info("Saving plot and clear axis & figure.")

        plt.axline((0, 0), (1, 1), linestyle='--', color='black')
        fig.savefig(path.join(dirpath, f'roc_auc.svg'))
        plt.clf()    # clear axis & figure so it does not affect the next plot.
