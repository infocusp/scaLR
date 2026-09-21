"""This is a test file for calibration.py"""

import numpy as np
import torch

from scalr.calibration import brier_score
from scalr.calibration import expected_calibration_error
from scalr.calibration import OpenSetThresholds
from scalr.calibration import predictive_entropy
from scalr.calibration import TemperatureScaler
from scalr.calibration import top1_top2_margin


def test_temperature_scaler_roundtrip():
    """A TemperatureScaler's dict form should reconstruct an equivalent scaler."""
    scaler = TemperatureScaler(temperature=2.5)
    d = scaler.to_dict()
    restored = TemperatureScaler.from_dict(d)
    assert restored.temperature == scaler.temperature


def test_temperature_scaler_fit_reduces_overconfidence():
    """Fitting on confident-but-frequently-wrong logits should push temperature above 1."""
    torch.manual_seed(0)
    n, n_classes = 300, 3
    labels = torch.randint(0, n_classes, (n,))

    # Model confidently predicts a class that is wrong 40% of the time:
    # classic overconfidence that temperature scaling should soften (T > 1).
    predicted_class = labels.clone()
    flip_idx = torch.randperm(n)[:int(n * 0.4)]
    predicted_class[flip_idx] = (labels[flip_idx] + 1) % n_classes

    logits = torch.randn(n, n_classes) * 0.1
    logits[torch.arange(n), predicted_class] += 8.0

    scaler = TemperatureScaler()
    scaler.fit(logits, labels)

    assert scaler.temperature > 1.0


def test_predictive_entropy_bounds():
    """Entropy should be ~0 for a one-hot distribution and ~1 for uniform."""
    one_hot = np.array([[1.0, 0.0, 0.0]])
    uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])

    assert predictive_entropy(one_hot)[0] < 1e-3
    assert predictive_entropy(uniform)[0] > 0.99


def test_top1_top2_margin():
    """Margin should equal the gap between the two largest probabilities."""
    probs = np.array([[0.6, 0.3, 0.1]])
    margin = top1_top2_margin(probs)
    assert np.isclose(margin[0], 0.3)


def test_expected_calibration_error_perfect_calibration():
    """ECE should be ~0 when confidence exactly matches accuracy."""
    probabilities = np.array([[0.9, 0.1]] * 10 + [[0.1, 0.9]] * 10)
    labels = np.array([0] * 9 + [1] + [1] * 9 + [0])
    ece = expected_calibration_error(probabilities, labels, n_bins=5)
    assert ece < 0.2


def test_brier_score_perfect_predictions():
    """Brier score should be 0 for confident, correct one-hot predictions."""
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    assert brier_score(probabilities, labels) == 0.0


def test_open_set_thresholds_flags_low_confidence():
    """A cell with low confidence, high entropy and low margin should be flagged unknown."""
    thresholds = OpenSetThresholds(min_confidence=0.5,
                                   max_entropy=0.7,
                                   min_margin=0.05)
    confidence = np.array([0.9, 0.3])
    entropy = np.array([0.1, 0.9])
    margin = np.array([0.8, 0.02])

    flags = thresholds.flag_unknown(confidence, entropy, margin)
    assert list(flags) == [False, True]


def test_open_set_thresholds_roundtrip():
    """An OpenSetThresholds' dict form should reconstruct an equivalent object."""
    thresholds = OpenSetThresholds(min_confidence=0.6, max_entropy=0.5,
                                   min_margin=0.1)
    restored = OpenSetThresholds.from_dict(thresholds.to_dict())
    assert restored == thresholds


def test_predictive_entropy_single_class_is_zero():
    """A single-class distribution has no uncertainty to measure; entropy should be ~0."""
    probs = np.array([[1.0]])
    assert abs(predictive_entropy(probs)[0]) < 1e-6


def test_top1_top2_margin_single_class_returns_top1():
    """With only one class, the margin degenerates to the top-1 probability."""
    probs = np.array([[0.8]])
    margin = top1_top2_margin(probs)
    assert np.isclose(margin[0], 0.8)


def test_expected_calibration_error_worst_case_is_bounded():
    """A maximally overconfident-and-wrong model should have high but <=1 ECE."""
    probabilities = np.array([[0.99, 0.01]] * 20)
    labels = np.array([1] * 20)    # always wrong despite high confidence.
    ece = expected_calibration_error(probabilities, labels, n_bins=10)
    assert 0.9 <= ece <= 1.0


def test_temperature_scaler_calibrate_probabilities_sum_to_one():
    """Calibrated probabilities should form a valid distribution per row."""
    scaler = TemperatureScaler(temperature=2.0)
    logits = np.array([[2.0, 1.0, 0.1], [0.5, 0.5, 0.5]])
    probs = scaler.calibrate_probabilities(logits)
    assert np.allclose(probs.sum(axis=1), 1.0)
