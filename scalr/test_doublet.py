"""This is a test file for doublet.py"""

import numpy as np

from scalr.calibration import predictive_entropy
from scalr.calibration import top1_top2_margin
from scalr.doublet import flag_possible_doublets


def test_flag_possible_doublets_flags_close_competing_classes():
    """A cell with two close, substantial class probabilities should be flagged."""
    probabilities = np.array([
        [0.48, 0.47, 0.05],    # ambiguous: two competing classes.
        [0.95, 0.03, 0.02],    # confident single class.
    ])
    entropy = predictive_entropy(probabilities)
    margin = top1_top2_margin(probabilities)

    flags = flag_possible_doublets(probabilities, entropy, margin)

    assert list(flags) == [True, False]


def test_flag_possible_doublets_low_second_probability_not_flagged():
    """A low-confidence prediction without a real second candidate should not be flagged."""
    probabilities = np.array([[0.4, 0.35, 0.25]])
    entropy = predictive_entropy(probabilities)
    margin = top1_top2_margin(probabilities)

    flags = flag_possible_doublets(probabilities,
                                   entropy,
                                   margin,
                                   min_second_probability=0.5)

    assert not flags[0]


def test_flag_possible_doublets_single_class_never_flags():
    """With only one class, there is no competing class to flag as a doublet."""
    probabilities = np.array([[1.0]])
    entropy = np.zeros(1)
    margin = np.ones(1)

    flags = flag_possible_doublets(probabilities, entropy, margin)
    assert list(flags) == [False]
