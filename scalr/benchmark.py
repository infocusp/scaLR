"""This file implements a lightweight, reproducible benchmark harness.

Reports scientific quality (macro-F1, balanced accuracy) alongside
computational cost (wall time, throughput, peak memory) for `scalr.train` +
`predict` runs across one or more datasets, plus the environment metadata
needed to make a run reproducible. This is a starting point for the "P1
benchmark suite" — enough to produce scaling-curve-able numbers without a
separate configuration format.
"""

from dataclasses import asdict
from dataclasses import dataclass
import platform
import resource
import sys
import time
from typing import Optional

from anndata import AnnData
import pandas as pd
import torch

import scalr


def peak_rss_mb() -> float:
    """Peak resident set size of this process so far, in MB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is kilobytes on Linux, bytes on macOS.
    return peak / 1024 if sys.platform != 'darwin' else peak / (1024 * 1024)


def capture_environment_metadata() -> dict:
    """Capture environment metadata for reproducible benchmark reporting."""
    return {
        'scalr_version': getattr(scalr, '__version__', 'unknown'),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'platform': platform.platform(),
    }


@dataclass
class BenchmarkResult:
    """A single benchmark run's quality + resource metrics."""

    name: str
    n_cells: int
    n_genes: int
    n_classes: int
    train_seconds: float
    predict_seconds: float
    cells_per_second_predict: float
    peak_rss_mb: float
    macro_f1: Optional[float]
    balanced_accuracy: Optional[float]


def run_benchmark(
    datasets: dict[str, AnnData],
    labels_key: str,
    group_key: Optional[str] = None,
    train_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """Benchmark scaLR train+predict quality and resource cost across datasets.

    Args:
        datasets: Mapping of benchmark name (e.g. 'small', 'medium', 'large')
            to an AnnData for that configuration.
        labels_key: Classification label column, shared across all datasets.
        group_key: Optional donor/patient/sample column for leakage-safe splitting.
        train_kwargs: Extra kwargs forwarded to `scalr.train` (e.g. `epochs`, `device`).

    Returns:
        A DataFrame indexed by dataset name, with one row of quality +
        resource-cost metrics per dataset — enough rows to plot a scaling
        curve across dataset sizes.
    """
    train_kwargs = dict(train_kwargs or {})
    rows = []

    for name, adata in datasets.items():
        start = time.perf_counter()
        model = scalr.train(adata,
                            labels_key=labels_key,
                            group_key=group_key,
                            verbose=False,
                            **train_kwargs)
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        result = model.predict(adata, device=train_kwargs.get('device', 'auto'))
        predict_seconds = time.perf_counter() - start

        rows.append(
            BenchmarkResult(
                name=name,
                n_cells=adata.shape[0],
                n_genes=adata.shape[1],
                n_classes=len(model.class_names),
                train_seconds=train_seconds,
                predict_seconds=predict_seconds,
                cells_per_second_predict=(adata.shape[0] /
                                          predict_seconds if predict_seconds > 0
                                          else float('inf')),
                peak_rss_mb=peak_rss_mb(),
                macro_f1=model.metrics.get('macro_f1'),
                balanced_accuracy=model.metrics.get('balanced_accuracy'),
            ))

    return pd.DataFrame([asdict(r) for r in rows]).set_index('name')
