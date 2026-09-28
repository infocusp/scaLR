"""This is a test file for metrics.py"""

import numpy as np
import pandas as pd

from scalr.metrics import check_class_imbalance
from scalr.metrics import compute_class_weights
from scalr.metrics import compute_classification_metrics


def test_compute_classification_metrics_perfect_predictions():
    """Perfect predictions should give macro-F1 and balanced accuracy of 1."""
    labels = np.array([0, 0, 1, 1, 2, 2])
    predictions = np.array([0, 0, 1, 1, 2, 2])
    class_names = ['A', 'B', 'C']

    metrics = compute_classification_metrics(labels, predictions, class_names)

    assert metrics.macro_f1 == 1.0
    assert metrics.weighted_f1 == 1.0
    assert metrics.balanced_accuracy == 1.0
    assert metrics.per_class.loc['A', 'support'] == 2


def test_compute_classification_metrics_rare_class_recall():
    """A rare, always-misclassified class should show up with low recall."""
    # Class 'C' (rare) is always predicted as 'A'.
    labels = np.array([0] * 20 + [1] * 20 + [2] * 2)
    predictions = np.array([0] * 20 + [1] * 20 + [0] * 2)
    class_names = ['A', 'B', 'C']

    metrics = compute_classification_metrics(labels, predictions, class_names)

    assert metrics.per_class.loc['C', 'recall'] == 0.0
    # Macro-F1 penalizes the rare-class failure much more than accuracy would.
    assert metrics.macro_f1 < 0.9


def test_check_class_imbalance_flags_small_classes():
    """A severely imbalanced label distribution should flag the small class."""
    labels = pd.Series(['T_cell'] * 1000 + ['DC'] * 5)
    report = check_class_imbalance(labels, min_class_size=50)

    assert 'DC' in report.small_classes
    assert report.imbalance_ratio > 1


def test_compute_class_weights_inverse_frequency():
    """Rarer classes should get larger weights than common classes."""
    labels = np.array([0] * 90 + [1] * 10)
    weights = compute_class_weights(labels, n_classes=2)

    assert weights[1] > weights[0]


def test_compute_classification_metrics_class_with_zero_support():
    """A class absent from the evaluation set should show support=0, not crash."""
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0, 0, 1, 1])
    class_names = ['A', 'B', 'Unseen']

    metrics = compute_classification_metrics(labels, predictions, class_names)

    assert metrics.per_class.loc['Unseen', 'support'] == 0
    assert metrics.confusion.shape == (3, 3)


def test_rare_class_recall_filters_by_support():
    """rare_class_recall should return only classes below the given support threshold."""
    labels = np.array([0] * 50 + [1] * 3)
    predictions = np.array([0] * 50 + [1] * 3)
    class_names = ['common', 'rare']

    metrics = compute_classification_metrics(labels, predictions, class_names)
    rare = metrics.rare_class_recall(min_support=10)

    assert list(rare.index) == ['rare']


def test_check_class_imbalance_balanced_data_flags_nothing():
    """A roughly balanced label distribution should not flag any small classes."""
    labels = pd.Series(['A'] * 100 + ['B'] * 100)
    report = check_class_imbalance(labels, min_class_size=10)

    assert report.small_classes == []
    assert report.imbalance_ratio == 1.0


def test_compute_class_weights_balanced_labels_are_equal():
    """Class weights should be equal (1.0) when classes are perfectly balanced."""
    labels = np.array([0] * 50 + [1] * 50)
    weights = compute_class_weights(labels, n_classes=2)
    assert np.allclose(weights, [1.0, 1.0])
