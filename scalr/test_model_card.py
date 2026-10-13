"""This is a test file for model_card.py"""

from scalr.model_card import render_model_card


def test_render_model_card_includes_key_fields():
    """The rendered card should surface labels, version, and metrics."""
    metadata = {
        'model_name': 'human_pbmc',
        'model_version': '1.0.0',
        'scalr_version': '2.0.0.dev0',
        'species': 'human',
        'random_seed': 42,
        'labels_key': 'cell_type',
        'group_key': 'donor_id',
        'split_ratio': [0.7, 0.1, 0.2],
        'feature_count': 5000,
        'labels': ['B_cell', 'T_cell', 'DC'],
        'temperature': 1.3,
        'open_set_thresholds': {
            'min_confidence': 0.5
        },
    }
    metrics = {
        'macro_f1': 0.812345,
        'weighted_f1': 0.9,
        'balanced_accuracy': 0.85,
        'n_test_cells': 1200,
    }

    card = render_model_card(metadata, metrics)

    assert 'human_pbmc' in card
    assert 'donor_id' in card
    assert 'B_cell, T_cell, DC' in card
    assert '0.8123' in card
    assert 'T=1.300' in card


def test_render_model_card_handles_missing_metrics():
    """A model card with no recorded metrics should still render without error."""
    metadata = {'labels_key': 'cell_type', 'labels': ['A', 'B']}
    card = render_model_card(metadata, {})
    assert 'No held-out metrics recorded.' in card


def test_render_model_card_handles_missing_group_key():
    """A model trained without group_key should say so explicitly."""
    metadata = {
        'labels_key': 'cell_type',
        'labels': ['A', 'B'],
        'group_key': None
    }
    card = render_model_card(metadata, {})
    assert 'none (no leakage-safe grouping used)' in card
