"""This file implements a doublet/mixed-cell screening signal.

A classifier should not silently force an ambiguous, likely-mixed cell into a
single lineage. This module flags cells whose top-two class probabilities are
both substantial and close together as `POSSIBLE_DOUBLET` candidates.

This is a screening signal derived from prediction uncertainty, not a
validated doublet caller (e.g. Scrublet/DoubletFinder) — treat it as "worth a
second look", not a definitive biological diagnosis.
"""

import numpy as np


def flag_possible_doublets(
    probabilities: np.ndarray,
    entropy: np.ndarray,
    margin: np.ndarray,
    max_margin: float = 0.15,
    min_entropy: float = 0.5,
    min_second_probability: float = 0.25,
) -> np.ndarray:
    """Flag cells with two closely-competing, substantial class probabilities.

    Args:
        probabilities: [n_cells, n_classes] calibrated class probabilities.
        entropy: Normalized predictive entropy per cell (see `scalr.calibration`).
        margin: Top1-top2 probability margin per cell.
        max_margin: A cell is only flagged if its margin is <= this.
        min_entropy: A cell is only flagged if its entropy is >= this.
        min_second_probability: A cell is only flagged if its second-highest
            class probability is >= this (i.e. a real competing signal, not
            just low confidence in a single class).

    Returns:
        Boolean array, True where the prediction looks like a possible
        mixed/doublet cell.
    """
    if probabilities.shape[1] < 2:
        return np.zeros(probabilities.shape[0], dtype=bool)

    sorted_probs = np.sort(probabilities, axis=1)
    second_probability = sorted_probs[:, -2]

    return ((margin <= max_margin) & (entropy >= min_entropy) &
            (second_probability >= min_second_probability))
