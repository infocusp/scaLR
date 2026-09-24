"""This file implements cluster-aware post-processing refinement of predictions.

Per-cell classifier predictions can be noisy within a transcriptionally
homogeneous cluster. This module optionally relabels cells to their cluster's
confidence-weighted majority class, while always preserving the original,
unrefined predictions rather than silently overwriting them.
"""

from dataclasses import replace
from typing import Optional

from anndata import AnnData
import numpy as np
import pandas as pd

from scalr.result import PredictionResult

_DEFAULT_CLUSTER_KEYS = ('leiden', 'louvain', 'cluster', 'clusters')


def detect_cluster_key(adata: AnnData) -> Optional[str]:
    """Return the first column in `adata.obs` that looks like a cluster assignment, if any."""
    for key in _DEFAULT_CLUSTER_KEYS:
        if key in adata.obs.columns:
            return key
    return None


def refine_with_clusters(
    result: PredictionResult,
    adata: AnnData,
    cluster_key: str = 'auto',
    min_cluster_agreement: float = 0.5,
) -> PredictionResult:
    """Refine predictions using per-cluster, confidence-weighted majority vote.

    For each cluster, if a class holds at least `min_cluster_agreement` of
    that cluster's total confidence-weighted vote, every cell in the cluster
    is relabeled to that class. This never mutates `result`; it returns a new
    `PredictionResult` whose `labels` are the refined labels, with
    `metadata['raw_labels']` preserving the original, unrefined predictions
    so both are always available.

    Args:
        result: A `PredictionResult` whose `obs_names` match `adata.obs_names`.
        adata: AnnData carrying a cluster assignment column in `.obs`.
        cluster_key: Column name in `adata.obs`, or 'auto' to detect one of
            'leiden'/'louvain'/'cluster'/'clusters'.
        min_cluster_agreement: Minimum confidence-weighted vote share (0-1)
            required for a cluster's majority class to be applied.

    Returns:
        A new `PredictionResult` with refined `labels` and
        `metadata['raw_labels']` holding the pre-refinement labels.
    """
    if list(adata.obs_names) != list(result.obs_names):
        raise ValueError(
            'adata.obs_names must match result.obs_names for cluster refinement.'
        )

    if cluster_key == 'auto':
        detected = detect_cluster_key(adata)
        if detected is None:
            raise ValueError(
                'No cluster column found automatically (looked for '
                f'{_DEFAULT_CLUSTER_KEYS}); pass `cluster_key` explicitly.')
        cluster_key = detected
    elif cluster_key not in adata.obs.columns:
        raise ValueError(f'cluster_key="{cluster_key}" not found in adata.obs.')

    clusters = adata.obs[cluster_key].astype(str).values
    labels = np.array(result.labels)
    confidence = np.asarray(result.confidence)

    refined = labels.copy()
    for cluster_id in np.unique(clusters):
        mask = clusters == cluster_id
        weights = pd.Series(confidence[mask]).groupby(labels[mask]).sum()
        total_weight = weights.sum()
        if total_weight == 0:
            continue
        majority_label = weights.idxmax()
        if weights[majority_label] / total_weight >= min_cluster_agreement:
            refined[mask] = majority_label

    metadata = dict(result.metadata)
    metadata['raw_labels'] = labels.tolist()
    metadata['cluster_key'] = cluster_key

    return replace(result, labels=refined.tolist(), metadata=metadata)
