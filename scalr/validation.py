"""This file implements the AnnData input-validation subsystem.

It replaces silent downstream failures (obscure numpy/torch/sklearn tracebacks)
with an explicit, user-facing validation report.
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Optional, Union

from anndata import AnnData
import numpy as np


@dataclass
class ValidationReport:
    """Structured validation result with error/warning/info messages.

    `errors` block usage of the data (`is_usable` is False when non-empty).
    `warnings` describe issues that should be surfaced but do not block usage.
    `info` records descriptive facts about the input.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def add_info(self, message: str) -> None:
        self.info.append(message)

    @property
    def is_usable(self) -> bool:
        return len(self.errors) == 0

    def raise_if_unusable(self) -> None:
        if not self.is_usable:
            raise ValueError('scaLR validation failed:\n' +
                             '\n'.join(f'  - {e}' for e in self.errors))

    def __str__(self) -> str:
        lines = ['scaLR validation', '-' * 17]
        lines.append('ERROR')
        lines += [f'  {e}' for e in self.errors] if self.errors else ['  None']
        lines.append('')
        lines.append('WARNING')
        lines += [f'  {w}' for w in self.warnings
                 ] if self.warnings else ['  None']
        lines.append('')
        lines.append('INFO')
        lines += [f'  {i}' for i in self.info] if self.info else ['  None']
        lines.append('')
        status = 'usable' if self.is_usable else 'unusable'
        if self.is_usable and self.warnings:
            status += ' with warning'
        lines.append(f'Status: {status}')
        return '\n'.join(lines)


def detect_normalization_state(adata: AnnData,
                               layer: Optional[str] = None,
                               sample_size: int = 500) -> str:
    """Heuristically detect whether expression values look like raw counts,
    log1p-normalized values, or standardized/scaled values.

    This is a heuristic, not a guarantee. It is used to inform the user (and
    `preprocess="auto"`); it never silently transforms the data.

    Returns:
        One of 'raw_counts', 'log1p_normalized', 'scaled', 'unknown'.
    """
    X = adata.layers[layer] if layer else adata.X
    n = min(sample_size, X.shape[0])
    if n == 0:
        return 'unknown'
    sample = X[:n]
    if not isinstance(sample, np.ndarray):
        sample = sample.toarray()
    sample = np.asarray(sample, dtype=np.float64)

    if sample.size == 0:
        return 'unknown'

    has_negative = (sample < 0).any()
    if has_negative:
        return 'scaled'

    is_integer_like = np.allclose(sample, np.round(sample), atol=1e-6)
    if is_integer_like and sample.max() > 30:
        return 'raw_counts'

    if sample.max() < 30 and not is_integer_like:
        return 'log1p_normalized'

    return 'unknown'


def validate(
    adata: AnnData,
    labels_key: Optional[str] = None,
    group_key: Optional[str] = None,
    model_features: Optional[list[str]] = None,
    min_feature_overlap: float = 0.1,
) -> ValidationReport:
    """Run the scaLR input-validation subsystem on an AnnData object.

    Args:
        adata: Input AnnData object.
        labels_key: Column in `adata.obs` expected to hold classification labels
            (checked during training-style validation).
        group_key: Column in `adata.obs` expected to hold a grouping identifier
            such as donor/patient/sample (checked when leakage-safe splitting
            is required).
        model_features: Ordered gene/feature list of a reference model, used to
            report gene-overlap coverage against the query `adata`.
        min_feature_overlap: Minimum fraction of `model_features` that must be
            present in `adata.var_names` for the data to remain usable.

    Returns:
        A `ValidationReport` describing structural, numerical and metadata
        issues found in `adata`.
    """
    report = ValidationReport()

    if not isinstance(adata, AnnData):
        report.add_error(
            f'Expected an AnnData object, got {type(adata).__name__}.')
        return report

    # --- Data structure checks -------------------------------------------------
    n_obs, n_vars = adata.shape
    if n_obs == 0:
        report.add_error('AnnData has 0 observations (cells).')
    if n_vars == 0:
        report.add_error('AnnData has 0 variables (genes).')

    if adata.obs_names.duplicated().any():
        n_dup = int(adata.obs_names.duplicated().sum())
        report.add_error(f'{n_dup} duplicate cell IDs (obs_names) found.')

    if adata.var_names.duplicated().any():
        n_dup = int(adata.var_names.duplicated().sum())
        report.add_warning(
            f'{n_dup} duplicate gene IDs (var_names) found; only the first '
            'occurrence will be used during gene alignment.')

    if not report.is_usable:
        return report

    # --- Numerical checks --------------------------------------------------
    X = adata.X
    sample_n = min(2000, n_obs)
    X_sample = X[:sample_n]
    if not isinstance(X_sample, np.ndarray):
        X_sample = X_sample.toarray()
    X_sample = np.asarray(X_sample)

    if np.isnan(X_sample).any():
        report.add_error('NaN values detected in `adata.X`.')
    if np.isinf(X_sample).any():
        report.add_error('Inf values detected in `adata.X`.')

    if not report.is_usable:
        return report

    library_sizes = np.asarray(X_sample.sum(axis=1)).flatten()
    n_zero_count = int((library_sizes == 0).sum())
    if n_zero_count:
        report.add_warning(
            f'{n_zero_count}/{sample_n} sampled cells have zero total counts.')

    positive_library_sizes = library_sizes[library_sizes > 0]
    if positive_library_sizes.size:
        huge = library_sizes > (np.median(positive_library_sizes) * 100)
        if huge.any():
            report.add_warning(
                f'{int(huge.sum())}/{sample_n} sampled cells have unusually '
                'large library sizes (>100x the sample median).')

    norm_state = detect_normalization_state(adata)
    report.add_info(f'Detected expression state: {norm_state}')

    # --- Metadata checks -----------------------------------------------------
    if labels_key is not None:
        if labels_key not in adata.obs.columns:
            report.add_error(
                f'labels_key="{labels_key}" not found in adata.obs columns.')
        else:
            n_missing = int(adata.obs[labels_key].isna().sum())
            if n_missing:
                report.add_warning(
                    f'{n_missing} cells have a missing "{labels_key}" label.')
            class_counts = adata.obs[labels_key].value_counts()
            report.add_info(
                f'Label "{labels_key}": {class_counts.shape[0]} classes, '
                f'smallest class size={int(class_counts.min())}.')

    if group_key is not None and group_key not in adata.obs.columns:
        report.add_error(
            f'group_key="{group_key}" not found in adata.obs columns.')

    # --- Model / gene-alignment compatibility ---------------------------------
    if model_features is not None:
        query_genes = set(adata.var_names)
        model_genes = set(model_features)
        matched = model_genes & query_genes
        coverage = len(matched) / len(model_genes) if model_genes else 0.0
        report.add_info(f'Input: {n_obs:,} cells x {n_vars:,} genes; '
                        f'model expects {len(model_genes):,} genes; '
                        f'gene coverage: {coverage * 100:.1f}%.')
        if len(model_genes) - len(matched) > 0:
            report.add_warning(
                f'{len(model_genes) - len(matched)}/{len(model_genes)} model '
                'genes are missing from the query.')
        if coverage < min_feature_overlap:
            report.add_error(
                f'Gene coverage ({coverage * 100:.1f}%) is below the minimum '
                f'required overlap ({min_feature_overlap * 100:.1f}%).')
    else:
        report.add_info(f'Input: {n_obs:,} cells x {n_vars:,} genes.')

    is_sparse = not isinstance(adata.X, np.ndarray)
    report.add_info(f'Sparse: {"yes" if is_sparse else "no"}')

    return report
