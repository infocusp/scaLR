"""This file implements dataset-level drift detection: whether a whole new
(query) dataset's gene-expression distribution looks meaningfully different
from the data a model was trained on.

Per-cell signals (`is_unknown` open-set abstention, `is_possible_doublet`,
see `scalr.calibration`/`scalr.doublet`) catch individual out-of-distribution
cells. This catches the case a per-cell signal can miss: a whole new cohort,
batch, tissue, or protocol that looks systematically different even if most
individual cells still score confidently, which matters most for a low-
resource classifier deployed on new, unseen cohorts.

Reference statistics (per-gene mean/std, per-cell library size and detected-
gene-count distributions) are computed once at training time from the
training split and stored in the model artifact (`drift_reference.json`), so
drift detection at inference time never needs the original training data.
"""

from dataclasses import dataclass
from dataclasses import field

from anndata import AnnData
import numpy as np
from scipy import sparse


def _row_stats(X) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell (library_size, n_genes_detected), without densifying `X`."""
    if sparse.issparse(X):
        library_size = np.asarray(X.sum(axis=1)).ravel()
        n_genes_detected = np.asarray((X > 0).sum(axis=1)).ravel()
    else:
        X = np.asarray(X)
        library_size = X.sum(axis=1)
        n_genes_detected = (X > 0).sum(axis=1)
    return library_size.astype(np.float64), n_genes_detected.astype(np.float64)


def _column_mean_std(X) -> tuple[np.ndarray, np.ndarray]:
    """Per-gene (mean, std), without densifying `X`."""
    if sparse.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    else:
        X = np.asarray(X)
        mean = X.mean(axis=0)
        mean_sq = (X**2).mean(axis=0)
    std = np.sqrt(np.maximum(mean_sq - mean**2, 0.0))
    return mean, std


def compute_reference_stats(adata: AnnData) -> dict:
    """Compute per-gene and per-cell reference statistics from `adata`.

    `adata` is expected to already be restricted to the model's training
    cells and aligned to the model's feature order (e.g. the training split,
    in `features` order), so `gene_mean`/`gene_std` line up 1:1 with
    `AnnotationModel.features` at drift-detection time.

    Returns a JSON-serializable dict, suitable for `drift_reference.json` in
    a model artifact.
    """
    gene_mean, gene_std = _column_mean_std(adata.X)
    library_size, n_genes_detected = _row_stats(adata.X)

    return {
        'n_cells': int(adata.shape[0]),
        'gene_mean': gene_mean.tolist(),
        'gene_std': gene_std.tolist(),
        'library_size_mean': float(library_size.mean()),
        'library_size_std': float(library_size.std()),
        'n_genes_detected_mean': float(n_genes_detected.mean()),
        'n_genes_detected_std': float(n_genes_detected.std()),
    }


@dataclass
class DriftReport:
    """Comparison of a query dataset's distribution against a model's
    training reference statistics."""

    n_query_cells: int
    n_reference_cells: int
    n_features: int
    drifted_gene_fraction: float
    top_drifted_genes: list[tuple[str, float]] = field(default_factory=list)
    library_size_z: float = 0.0
    n_genes_detected_z: float = 0.0
    drift_fraction_threshold: float = 0.1
    severity_z_threshold: float = 5.0

    @property
    def is_drifted(self) -> bool:
        return (self.drifted_gene_fraction >= self.drift_fraction_threshold or
                abs(self.library_size_z) >= self.severity_z_threshold or
                abs(self.n_genes_detected_z) >= self.severity_z_threshold)

    def __str__(self) -> str:
        lines = [
            'scaLR drift report',
            '-' * 19,
            f'Query cells:                {self.n_query_cells:,}',
            f'Reference (training) cells: {self.n_reference_cells:,}',
            f'Drifted genes:              '
            f'{self.drifted_gene_fraction * 100:.1f}% of {self.n_features:,} '
            f'(threshold {self.drift_fraction_threshold * 100:.0f}%)',
            f'Library size shift (z):     {self.library_size_z:+.2f}',
            f'Genes-detected shift (z):   {self.n_genes_detected_z:+.2f}',
            '',
            'Top drifted genes (|z-score| of mean expression shift):',
        ]
        if self.top_drifted_genes:
            lines += [
                f'  {gene}: z={z:+.2f}' for gene, z in self.top_drifted_genes
            ]
        else:
            lines.append('  None')
        lines.append('')
        lines.append(f'Status: {"DRIFTED" if self.is_drifted else "OK"}')
        return '\n'.join(lines)


def detect_drift(
    query: AnnData,
    reference: dict,
    features: list[str],
    z_threshold: float = 2.0,
    drift_fraction_threshold: float = 0.1,
    severity_z_threshold: float = 5.0,
    top_k: int = 20,
) -> DriftReport:
    """Compare a gene-aligned query dataset against a model's training
    reference statistics.

    Args:
        query: Query AnnData, already gene-aligned to `features` (e.g. via
            `scalr.align_genes`), so its columns line up 1:1 with
            `reference['gene_mean']`/`reference['gene_std']`.
        reference: Reference stats dict, as produced by
            `compute_reference_stats` (stored in a model artifact as
            `drift_reference.json`).
        features: The model's ordered feature list, for labeling drifted
            genes by name.
        z_threshold: Per-gene standardized mean shift above which a gene
            counts as "drifted".
        drift_fraction_threshold: Fraction of drifted genes above which the
            dataset as a whole is flagged `is_drifted`.
        severity_z_threshold: A library-size or detected-gene-count shift
            this extreme (in reference std deviations) flags `is_drifted` on
            its own, even if per-gene drift looks mild — catches e.g. a
            sequencing-depth/protocol change a per-gene test can miss.
        top_k: Number of top drifted genes to report.

    Returns:
        A `DriftReport`.
    """
    ref_mean = np.asarray(reference['gene_mean'], dtype=np.float64)
    ref_std = np.asarray(reference['gene_std'], dtype=np.float64)
    if len(ref_mean) != len(features) or query.shape[1] != len(features):
        raise ValueError(
            'Reference stats/features/query column count mismatch '
            f'({len(ref_mean)} vs {len(features)} vs {query.shape[1]}); '
            'query must already be gene-aligned to `features`.')

    query_mean, _ = _column_mean_std(query.X)
    safe_std = np.where(ref_std > 1e-8, ref_std, 1e-8)
    gene_z = (query_mean - ref_mean) / safe_std

    drifted_mask = np.abs(gene_z) >= z_threshold
    drifted_fraction = float(drifted_mask.mean()) if len(gene_z) else 0.0

    order = np.argsort(-np.abs(gene_z))[:top_k]
    top_drifted = [(features[i], float(gene_z[i])) for i in order]

    library_size, n_genes_detected = _row_stats(query.X)
    ref_lib_std = reference.get('library_size_std') or 1e-8
    ref_genes_std = reference.get('n_genes_detected_std') or 1e-8
    library_size_z = (library_size.mean() -
                      reference.get('library_size_mean', 0.0)) / ref_lib_std
    n_genes_detected_z = (n_genes_detected.mean() - reference.get(
        'n_genes_detected_mean', 0.0)) / ref_genes_std

    return DriftReport(
        n_query_cells=int(query.shape[0]),
        n_reference_cells=int(reference.get('n_cells', 0)),
        n_features=len(features),
        drifted_gene_fraction=drifted_fraction,
        top_drifted_genes=top_drifted,
        library_size_z=float(library_size_z),
        n_genes_detected_z=float(n_genes_detected_z),
        drift_fraction_threshold=drift_fraction_threshold,
        severity_z_threshold=severity_z_threshold,
    )
