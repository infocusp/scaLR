"""This file implements leakage-safe train/val/test splitting and leakage detection.

Splitting cells at random can put the same donor/patient/sample in both train
and test, letting a model "cheat" on donor-specific signal rather than
learning the classification task. This module makes group-safe splitting a
first-class, explicit operation.
"""

from dataclasses import dataclass
from dataclasses import field

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass
class LeakageReport:
    """Result of checking for group leakage across data splits."""

    group_key: str
    leaked_groups: dict = field(default_factory=dict)

    @property
    def has_leakage(self) -> bool:
        return len(self.leaked_groups) > 0

    def __str__(self) -> str:
        if not self.has_leakage:
            return f'No leakage detected across splits on "{self.group_key}".'
        lines = [
            f'Leakage detected: the following "{self.group_key}" values '
            'appear in more than one split:'
        ]
        for split_pair, groups in self.leaked_groups.items():
            lines.append(f'  {split_pair}: {sorted(groups)[:10]}'
                         f'{"..." if len(groups) > 10 else ""}')
        lines.append(
            'Recommendation: use group-safe splitting (split_strategy="group") '
            f'on column "{self.group_key}".')
        return '\n'.join(lines)


def check_group_leakage(splits: dict[str, pd.DataFrame],
                        group_key: str) -> LeakageReport:
    """Check whether any group value spans more than one split.

    Args:
        splits: Mapping of split name (e.g. 'train', 'val', 'test') to that
            split's `obs` DataFrame.
        group_key: Column name identifying donor/patient/sample/batch.

    Returns:
        A `LeakageReport`.
    """
    split_groups = {
        name: set(df[group_key].unique())
        for name, df in splits.items()
        if group_key in df.columns
    }
    names = list(split_groups)
    leaked = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = split_groups[names[i]] & split_groups[names[j]]
            if overlap:
                leaked[f'{names[i]} <-> {names[j]}'] = overlap

    return LeakageReport(group_key=group_key, leaked_groups=leaked)


def detect_likely_grouping_column(
        obs: pd.DataFrame,
        candidates: tuple[str, ...] = ('donor_id', 'donor', 'patient_id',
                                       'patient', 'sample_id', 'sample',
                                       'subject_id', 'batch')) -> list[str]:
    """Detect columns in `obs` that look like a donor/patient/sample identifier.

    This is a name-based heuristic used to warn users who did not specify a
    `group_key`, not a guarantee that grouped splitting is required.
    """
    return [c for c in candidates if c in obs.columns]


def group_safe_split(
    obs: pd.DataFrame,
    labels_key: str,
    group_key: str = None,
    split_ratio: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Split sample indices into train/val/test, keeping each group in a single split.

    Falls back to a stratified (non-grouped) split when `group_key` is None,
    but this should only be done when no donor/patient/sample identifier is
    available, since it risks train/test leakage.

    Args:
        obs: `adata.obs` DataFrame.
        labels_key: Column used for stratification.
        group_key: Column identifying donor/patient/sample; when given, a
            group never spans more than one split.
        split_ratio: (train, val, test) ratios; need not sum to 1.
        seed: Random seed for reproducibility.

    Returns:
        Dict with 'train', 'val', 'test' -> integer position arrays into `obs`.
    """
    n = len(obs)
    indices = np.arange(n)
    total = sum(split_ratio)
    train_r, val_r, test_r = (r / total for r in split_ratio)

    def _split(idx, y, groups, test_size):
        if groups is not None:
            splitter = GroupShuffleSplit(test_size=test_size,
                                         n_splits=1,
                                         random_state=seed)
            return next(splitter.split(idx, y[idx], groups=groups[idx]))
        splitter = StratifiedShuffleSplit(test_size=test_size,
                                          n_splits=1,
                                          random_state=seed)
        return next(splitter.split(idx, y[idx]))

    y = obs[labels_key].values
    groups = obs[group_key].values if group_key else None

    train_val_pos, test_pos = _split(indices, y, groups, test_ratio := test_r)
    train_val_idx = indices[train_val_pos]
    test_idx = indices[test_pos]

    relative_val_ratio = val_r / (train_r + val_r)
    train_pos, val_pos = _split(train_val_idx, y, groups, relative_val_ratio)
    train_idx = train_val_idx[train_pos]
    val_idx = train_val_idx[val_pos]

    return {'train': train_idx, 'val': val_idx, 'test': test_idx}
