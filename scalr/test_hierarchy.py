"""This is a test file for hierarchy.py"""

import numpy as np
import pytest

from scalr.hierarchy import aggregate_to_broad
from scalr.hierarchy import validate_taxonomy
from scalr.result import PredictionResult


def _fine_result():
    class_names = ['CD4_T', 'CD8_T', 'B_cell']
    probabilities = np.array([
        [0.6, 0.3, 0.1],
        [0.1, 0.1, 0.8],
        [0.2, 0.7, 0.1],
    ])
    return PredictionResult(
        obs_names=['c0', 'c1', 'c2'],
        labels=['CD4_T', 'B_cell', 'CD8_T'],
        probabilities=probabilities,
        class_names=class_names,
        confidence=probabilities.max(axis=1),
        entropy=np.zeros(3),
        margin=np.full(3, 0.3),
        is_unknown=np.array([False, False, True]),
    )


def test_validate_taxonomy_passes_with_full_coverage():
    """A taxonomy covering every class should validate without error."""
    validate_taxonomy(['A', 'B'], {'A': 'x', 'B': 'y'})


def test_validate_taxonomy_raises_on_missing_class():
    """A taxonomy missing an entry for a known class should raise."""
    with pytest.raises(ValueError):
        validate_taxonomy(['A', 'B'], {'A': 'x'})


def test_aggregate_to_broad_sums_fine_probabilities():
    """Broad-class probability should equal the sum of its fine-class probabilities."""
    result = _fine_result()
    taxonomy = {'CD4_T': 'T_cell', 'CD8_T': 'T_cell', 'B_cell': 'B_cell'}

    broad = aggregate_to_broad(result, taxonomy)

    assert sorted(broad.class_names) == ['B_cell', 'T_cell']
    t_cell_idx = broad.class_names.index('T_cell')
    # Cell 0: CD4_T (0.6) + CD8_T (0.3) = 0.9 T_cell probability.
    assert np.isclose(broad.probabilities[0, t_cell_idx], 0.9)


def test_aggregate_to_broad_preserves_obs_names_and_unknown():
    """Aggregation should not change obs_names or the fine-level unknown flag."""
    result = _fine_result()
    taxonomy = {'CD4_T': 'T_cell', 'CD8_T': 'T_cell', 'B_cell': 'B_cell'}

    broad = aggregate_to_broad(result, taxonomy)

    assert broad.obs_names == result.obs_names
    assert list(broad.is_unknown) == list(result.is_unknown)
    assert broad.metadata['level'] == 'broad'
    assert broad.metadata['fine_class_names'] == result.class_names


def test_aggregate_to_broad_raises_on_incomplete_taxonomy():
    """Aggregating with a taxonomy missing a fine class should raise."""
    result = _fine_result()
    with pytest.raises(ValueError):
        aggregate_to_broad(result, {'CD4_T': 'T_cell'})
