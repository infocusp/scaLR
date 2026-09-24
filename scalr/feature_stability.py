"""This file implements feature-selection stability analysis.

Feature selection (biomarker discovery) is only scientifically defensible if
the selected genes are stable across independent resamplings of the data,
not an artifact of one particular train/test split. This module repeats
feature selection on random subsamples and reports how consistently each
gene is selected, and how similar the selected sets are pairwise (Jaccard).
"""

from dataclasses import dataclass
from dataclasses import field

from anndata import AnnData
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class FeatureStabilityReport:
    """Stability of a repeated top-k feature-selection procedure."""

    selection_frequency: pd.Series
    n_runs: int
    top_k: int
    mean_jaccard: float

    def top_stable_features(self, min_runs: int = None) -> list[str]:
        """Genes selected in at least `min_runs` of the `n_runs` runs (default: half)."""
        min_runs = min_runs if min_runs is not None else max(
            self.n_runs // 2, 1)
        return self.selection_frequency[self.selection_frequency >=
                                        min_runs].index.tolist()

    def __str__(self) -> str:
        lines = [
            f'Feature-selection stability over {self.n_runs} runs (top-{self.top_k}):',
            '',
            f'{"Gene":<20}Selected runs / {self.n_runs}',
        ]
        for gene, count in self.selection_frequency.head(20).items():
            lines.append(f'{gene:<20}{count}')
        lines.append('')
        lines.append(
            f'Mean pairwise Jaccard similarity: {self.mean_jaccard:.3f}')
        return '\n'.join(lines)


def compute_feature_stability(
    adata: AnnData,
    labels_key: str,
    top_k: int = 50,
    n_runs: int = 20,
    subsample_frac: float = 0.8,
    seed: int = 42,
) -> FeatureStabilityReport:
    """Run repeated feature selection on random subsamples and measure stability.

    Each run fits a multinomial logistic regression on a random
    `subsample_frac` fraction of cells (sampled without replacement) and
    scores genes by the sum of `|coefficient|` across classes; the top-`k`
    genes by that score form the run's selected set.

    Args:
        adata: AnnData with normalized `adata.X` and label column `labels_key`.
        labels_key: Column in `adata.obs` with classification labels.
        top_k: Number of top-scoring genes selected per run.
        n_runs: Number of independent subsampling runs.
        subsample_frac: Fraction of cells sampled (without replacement) per run.
        seed: Base random seed; run `i` uses seed `seed + i`.

    Returns:
        A `FeatureStabilityReport` with per-gene selection frequency and the
        mean pairwise Jaccard similarity of the selected sets across runs.
    """
    genes = np.asarray(adata.var_names)
    n_cells = adata.shape[0]
    labels = adata.obs[labels_key].astype(str).values

    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    selected_sets = []
    gene_counts = pd.Series(0, index=genes)

    sample_size = max(int(n_cells * subsample_frac), 2)
    for run in range(n_runs):
        run_rng = np.random.default_rng(seed + run)
        idx = run_rng.choice(n_cells, size=sample_size, replace=False)

        clf = LogisticRegression(max_iter=200)
        clf.fit(X[idx], labels[idx])

        scores = np.abs(clf.coef_).sum(axis=0)
        top_idx = np.argsort(-scores)[:top_k]
        top_genes = set(genes[top_idx])
        selected_sets.append(top_genes)
        gene_counts[list(top_genes)] += 1

    gene_counts = gene_counts[gene_counts > 0].sort_values(ascending=False)

    jaccards = []
    for i in range(len(selected_sets)):
        for j in range(i + 1, len(selected_sets)):
            union = len(selected_sets[i] | selected_sets[j])
            intersection = len(selected_sets[i] & selected_sets[j])
            jaccards.append(intersection / union if union else 0.0)
    mean_jaccard = float(np.mean(jaccards)) if jaccards else 1.0

    return FeatureStabilityReport(
        selection_frequency=gene_counts,
        n_runs=n_runs,
        top_k=top_k,
        mean_jaccard=mean_jaccard,
    )
