"""This file implements class-imbalance-aware evaluation metrics."""

from dataclasses import dataclass
from dataclasses import field

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support


@dataclass
class ClassificationMetrics:
    """Class-imbalance-aware classification metrics.

    Overall accuracy is not reported as the headline metric because it is
    dominated by majority classes; macro-F1 and balanced accuracy weight
    every class equally so rare-cell performance is visible.
    """

    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    per_class: pd.DataFrame
    confusion: np.ndarray
    class_names: list[str]

    def rare_class_recall(self, min_support: int) -> pd.DataFrame:
        """Recall restricted to classes with support < `min_support` in the evaluation set."""
        return self.per_class[self.per_class['support'] < min_support]

    def __str__(self) -> str:
        lines = [
            f'macro-F1:          {self.macro_f1:.4f}',
            f'weighted-F1:       {self.weighted_f1:.4f}',
            f'balanced accuracy: {self.balanced_accuracy:.4f}',
        ]
        return '\n'.join(lines)


def compute_classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> ClassificationMetrics:
    """Compute macro/weighted-F1, balanced accuracy, and a per-class report.

    Args:
        labels: True integer class labels, shape [n_samples].
        predictions: Predicted integer class labels, shape [n_samples].
        class_names: Ordered class names corresponding to label ids [0..n_classes).

    Returns:
        A `ClassificationMetrics` object.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    all_ids = list(range(len(class_names)))

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=all_ids, zero_division=0)

    per_class = pd.DataFrame(
        {
            'class': class_names,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
        }).set_index('class')

    macro_f1 = f1_score(labels,
                        predictions,
                        labels=all_ids,
                        average='macro',
                        zero_division=0)
    weighted_f1 = f1_score(labels,
                           predictions,
                           labels=all_ids,
                           average='weighted',
                           zero_division=0)
    balanced_acc = balanced_accuracy_score(labels, predictions)
    conf = confusion_matrix(labels, predictions, labels=all_ids)

    return ClassificationMetrics(
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        balanced_accuracy=float(balanced_acc),
        per_class=per_class,
        confusion=conf,
        class_names=class_names,
    )


@dataclass
class ClassImbalanceReport:
    """Report describing class-size imbalance in a labeled dataset."""

    class_counts: pd.Series
    imbalance_ratio: float
    small_classes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ['Class imbalance detected' if self.small_classes else
                 'Class distribution']
        lines.append('')
        for cls, count in self.class_counts.items():
            lines.append(f'{cls}: {count:,}')
        lines.append('')
        if self.small_classes:
            lines.append(
                'Recommendation: enable class-balanced sampling or class-weighted loss.'
            )
        return '\n'.join(lines)


def check_class_imbalance(labels: pd.Series,
                          min_class_size: int = 50,
                          imbalance_ratio_threshold: float = 20.0
                         ) -> ClassImbalanceReport:
    """Report class sizes and flag rare classes / severe imbalance.

    Args:
        labels: Categorical label series (e.g. `adata.obs[labels_key]`).
        min_class_size: Classes with fewer samples than this are flagged as small.
        imbalance_ratio_threshold: max_class_size / min_class_size above which the
            dataset is considered severely imbalanced.

    Returns:
        A `ClassImbalanceReport`.
    """
    counts = labels.value_counts()
    ratio = float(counts.max() / max(counts.min(), 1))
    small = counts[counts < min_class_size].index.tolist()

    return ClassImbalanceReport(class_counts=counts,
                                imbalance_ratio=ratio,
                                small_classes=small)


def compute_class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse-frequency class weights for a weighted cross-entropy loss."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return weights
