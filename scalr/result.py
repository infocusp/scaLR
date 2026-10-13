"""This file defines the typed prediction result contract returned by the public API."""

from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PredictionResult:
    """Typed container for annotation/prediction outputs.

    Row order and identity always match the `obs_names` of the AnnData object
    that was passed to `predict`/`annotate`, so predictions cannot silently
    become misaligned with the input data.

    Attributes:
        obs_names: Cell identifiers, in the same order as all other arrays.
        labels: Top-1 predicted label (string) for each cell.
        probabilities: [n_cells, n_classes] calibrated class probabilities.
        class_names: Ordered class names corresponding to `probabilities` columns.
        confidence: Calibrated top-1 confidence for each cell.
        entropy: Predictive (normalized) entropy for each cell.
        margin: Top-1 minus top-2 probability margin for each cell.
        top_k: List of (label, probability) tuples per cell, top-k only.
        is_unknown: Whether the cell was flagged as an open-set/unknown prediction.
        ood_score: Optional out-of-distribution score (higher = more OOD).
        explanations: Optional per-cell explanation payload.
        metadata: Model/preprocessing/run metadata used to produce this result.
        is_possible_doublet: Optional screening flag for cells whose top-two
            class probabilities are both substantial and close together —
            a heuristic signal, not a validated doublet call.
    """

    obs_names: list[str]
    labels: list[str]
    probabilities: np.ndarray
    class_names: list[str]
    confidence: np.ndarray
    entropy: np.ndarray
    margin: np.ndarray
    is_unknown: np.ndarray
    top_k: list[list[tuple]] = field(default_factory=list)
    ood_score: Optional[np.ndarray] = None
    explanations: Optional[dict] = None
    metadata: dict = field(default_factory=dict)
    is_possible_doublet: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.labels)

    def to_frame(self) -> pd.DataFrame:
        """Return a DataFrame indexed by `obs_names`, suitable for merging into `adata.obs`."""
        data = {
            'scalr_pred': self.labels,
            'scalr_confidence': self.confidence,
            'scalr_entropy': self.entropy,
            'scalr_margin': self.margin,
            'scalr_unknown': self.is_unknown,
        }
        if self.is_possible_doublet is not None:
            data['scalr_possible_doublet'] = self.is_possible_doublet
        return pd.DataFrame(data, index=self.obs_names)

    def write_to_adata(self, adata, prefix: str = 'scalr') -> None:
        """Write prediction fields onto an AnnData object's `.obs`/`.obsm`/`.uns`.

        Args:
            adata: AnnData object whose `obs_names` must match `self.obs_names`.
            prefix: Prefix used for all written columns/keys.
        """
        if list(adata.obs_names) != list(self.obs_names):
            raise ValueError(
                'adata.obs_names does not match the prediction result obs_names; '
                'refusing to write to avoid misaligned predictions.')

        adata.obs[f'{prefix}_pred'] = self.labels
        adata.obs[f'{prefix}_confidence'] = self.confidence
        adata.obs[f'{prefix}_entropy'] = self.entropy
        adata.obs[f'{prefix}_margin'] = self.margin
        adata.obs[f'{prefix}_unknown'] = self.is_unknown
        if self.is_possible_doublet is not None:
            adata.obs[f'{prefix}_possible_doublet'] = self.is_possible_doublet
        adata.obsm[f'{prefix}_probabilities'] = self.probabilities
        adata.uns[prefix] = self.metadata
