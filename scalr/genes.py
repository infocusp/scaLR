"""This file implements gene identifier alignment between a reference model and query data."""

from dataclasses import dataclass
from dataclasses import field
from typing import Union

from anndata import AnnData
import numpy as np
from scipy import sparse


@dataclass
class GeneAlignmentReport:
    """Summary of aligning a query AnnData's genes onto a reference feature list."""

    reference_features: int
    query_genes: int
    matched_features: int
    missing_features: list[str]
    extra_genes: int

    @property
    def coverage(self) -> float:
        return (self.matched_features /
                self.reference_features) if self.reference_features else 0.0

    @property
    def is_usable(self) -> bool:
        return self.coverage >= self._min_overlap

    def __str__(self) -> str:
        lines = [
            f'Reference features: {self.reference_features:,}',
            f'Query genes:        {self.query_genes:,}',
            f'Matched features:    {self.matched_features:,} '
            f'({self.coverage * 100:.1f}%)',
            f'Missing features:      {len(self.missing_features):,}',
            f'Extra query genes:  {self.extra_genes:,}',
            '',
            f'Status: {"usable" if self.is_usable else "unusable"}'
            f'{" with warning" if self.is_usable and self.missing_features else ""}',
        ]
        return '\n'.join(lines)


def align_genes(
    query: AnnData,
    reference_features: list[str],
    min_feature_overlap: float = 0.1,
) -> tuple[AnnData, GeneAlignmentReport]:
    """Align a query AnnData's genes onto an ordered reference feature list.

    Genes present in `reference_features` but missing from the query are
    added as zero-filled columns. Genes in the query not present in
    `reference_features` are dropped. The result has columns in exactly
    `reference_features` order, so it can be fed directly to a model trained
    on that feature list.

    Args:
        query: Query AnnData object.
        reference_features: Ordered list of gene names the reference model expects.
        min_feature_overlap: Minimum required fraction of `reference_features`
            present in the query for the result to be marked usable.

    Returns:
        (aligned_adata, report): AnnData with `reference_features` columns (in
        order), and a `GeneAlignmentReport` describing the alignment.
    """
    query_genes = list(query.var_names)
    query_index = {g: i for i, g in enumerate(query_genes)}

    matched = [g for g in reference_features if g in query_index]
    missing = [g for g in reference_features if g not in query_index]

    n_obs = query.shape[0]
    n_ref = len(reference_features)

    col_map = {g: i for i, g in enumerate(reference_features)}
    src_cols = [query_index[g] for g in matched]
    dst_cols = [col_map[g] for g in matched]

    X_query = query.X
    if not sparse.issparse(X_query):
        X_query = sparse.csr_matrix(X_query)
    X_query = X_query.tocsc()

    aligned = sparse.lil_matrix((n_obs, n_ref), dtype=X_query.dtype)
    if src_cols:
        aligned[:, dst_cols] = X_query[:, src_cols]
    aligned = aligned.tocsr()

    aligned_adata = AnnData(X=aligned, obs=query.obs.copy())
    aligned_adata.var_names = reference_features

    report = GeneAlignmentReport(
        reference_features=n_ref,
        query_genes=len(query_genes),
        matched_features=len(matched),
        missing_features=missing,
        extra_genes=len(query_genes) - len(matched),
    )
    report._min_overlap = min_feature_overlap

    return aligned_adata, report
