"""This file implements hierarchical (coarse-to-fine) cell-type annotation.

Rather than training a separate broad-level model, a fine-grained
`PredictionResult` is aggregated to a broad level by summing calibrated
fine-class probabilities within each broad group, using a user-supplied
`taxonomy` mapping (fine label -> broad label). This lets a single trained
model represent both "this is definitely Immune" (broad, high confidence)
and "which lymphoid subtype is less certain" (fine, lower confidence).
"""

import numpy as np

from scalr.calibration import predictive_entropy
from scalr.calibration import top1_top2_margin
from scalr.result import PredictionResult


def validate_taxonomy(class_names: list[str], taxonomy: dict[str, str]) -> None:
    """Raise if `taxonomy` does not cover every class in `class_names`."""
    missing = [c for c in class_names if c not in taxonomy]
    if missing:
        raise ValueError(
            f'taxonomy is missing an entry for fine-grained classes: {missing}')


def aggregate_to_broad(result: PredictionResult,
                       taxonomy: dict[str, str]) -> PredictionResult:
    """Aggregate a fine-grained `PredictionResult` to broad-level labels.

    Args:
        result: A fine-grained prediction result (e.g. CD4 T cell, CD8 T cell, ...).
        taxonomy: Mapping from every fine class name in `result.class_names` to
            its broad class name (e.g. {'CD4 T cell': 'T cell', ...}).

    Returns:
        A new `PredictionResult` at the broad level. Broad-class probabilities
        are the sum of the fine-class probabilities within each broad group;
        confidence/entropy/margin/is_unknown are recomputed at the broad level.
    """
    validate_taxonomy(result.class_names, taxonomy)

    broad_names = sorted(set(taxonomy.values()))
    broad_index = {b: i for i, b in enumerate(broad_names)}
    fine_to_broad_idx = np.array(
        [broad_index[taxonomy[c]] for c in result.class_names])

    n_cells = result.probabilities.shape[0]
    broad_probs = np.zeros((n_cells, len(broad_names)))
    for fine_idx, broad_idx in enumerate(fine_to_broad_idx):
        broad_probs[:, broad_idx] += result.probabilities[:, fine_idx]

    top1 = broad_probs.argmax(axis=1)
    labels = [broad_names[i] for i in top1]
    confidence = broad_probs.max(axis=1)
    entropy = predictive_entropy(broad_probs)
    margin = top1_top2_margin(broad_probs)

    k = min(5, len(broad_names))
    top_k_idx = np.argsort(-broad_probs, axis=1)[:, :k]
    top_k_out = [[(broad_names[c], float(broad_probs[row, c]))
                  for c in idx]
                 for row, idx in enumerate(top_k_idx)]

    metadata = dict(result.metadata)
    metadata['level'] = 'broad'
    metadata['fine_class_names'] = result.class_names

    return PredictionResult(
        obs_names=result.obs_names,
        labels=labels,
        probabilities=broad_probs,
        class_names=broad_names,
        confidence=confidence,
        entropy=entropy,
        margin=margin,
    # Cells flagged unknown at the fine level remain unknown at the broad
    # level; a fine-level abstention is still a lack of broad-level evidence.
        is_unknown=result.is_unknown.copy(),
        top_k=top_k_out,
        metadata=metadata,
    )
