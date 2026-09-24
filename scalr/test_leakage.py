"""This is a test file for leakage.py"""

import numpy as np
import pandas as pd

from scalr.leakage import check_group_leakage
from scalr.leakage import detect_likely_grouping_column
from scalr.leakage import group_safe_split
from scalr.utils import generate_dummy_dge_anndata


def test_group_safe_split_no_leakage():
    """No donor should appear in more than one split when group_key is used."""
    adata = generate_dummy_dge_anndata(n_donors=10, cell_replicate=5)
    obs = adata.obs

    split = group_safe_split(obs,
                             labels_key='cell_type',
                             group_key='donor_id',
                             split_ratio=(0.6, 0.2, 0.2))

    splits_obs = {name: obs.iloc[idx] for name, idx in split.items()}
    report = check_group_leakage(splits_obs, 'donor_id')

    assert not report.has_leakage
    # every sample should be assigned to exactly one split.
    total = sum(len(v) for v in split.values())
    assert total == len(obs)


def test_check_group_leakage_detects_overlap():
    """A donor present in two splits should be flagged as leakage."""
    train = pd.DataFrame({'donor_id': ['d1', 'd1', 'd2']})
    test = pd.DataFrame({'donor_id': ['d2', 'd3']})

    report = check_group_leakage({'train': train, 'test': test}, 'donor_id')

    assert report.has_leakage
    assert 'd2' in report.leaked_groups['train <-> test']


def test_detect_likely_grouping_column():
    """Columns with donor/patient/sample-like names should be detected."""
    obs = pd.DataFrame({'donor_id': ['d1'], 'random_col': [1]})
    detected = detect_likely_grouping_column(obs)
    assert detected == ['donor_id']


def test_detect_likely_grouping_column_none_found():
    """No warning-worthy columns should be detected when none look like an identifier."""
    obs = pd.DataFrame({'cell_type': ['T'], 'random_col': [1]})
    assert detect_likely_grouping_column(obs) == []


def test_group_safe_split_without_group_key_falls_back_to_stratified():
    """Without a group_key, splitting should still stratify by label and cover all rows."""
    adata = generate_dummy_dge_anndata(n_donors=6, cell_replicate=5)
    obs = adata.obs

    split = group_safe_split(obs,
                             labels_key='cell_type',
                             group_key=None,
                             split_ratio=(0.6, 0.2, 0.2))

    total = sum(len(v) for v in split.values())
    assert total == len(obs)
    # No overlap between splits.
    all_idx = np.concatenate(list(split.values()))
    assert len(all_idx) == len(set(all_idx.tolist()))


def test_check_group_leakage_no_group_column_in_split():
    """A split missing the group column entirely should not be compared or crash."""
    train = pd.DataFrame({'donor_id': ['d1', 'd2']})
    other = pd.DataFrame({'some_other_col': [1, 2]})

    report = check_group_leakage({'train': train, 'other': other}, 'donor_id')

    assert not report.has_leakage


def test_leakage_report_str_no_leakage():
    """The report string should clearly state when no leakage is detected."""
    train = pd.DataFrame({'donor_id': ['d1']})
    test = pd.DataFrame({'donor_id': ['d2']})
    report = check_group_leakage({'train': train, 'test': test}, 'donor_id')
    assert 'No leakage detected' in str(report)
