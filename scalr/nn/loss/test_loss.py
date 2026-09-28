"""This is a test file for _loss.py"""

import pytest
import torch

from scalr.nn.loss import build_loss_fn
from scalr.nn.loss import CustomLossBase
import scalr.nn.loss as loss_module


def test_build_loss_fn_resolves_builtin_torch_loss():
    """A name matching a torch.nn class should build that built-in loss."""
    loss_fn, loss_config = build_loss_fn({'name': 'CrossEntropyLoss'})

    assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)
    assert loss_config == {'name': 'CrossEntropyLoss'}


def test_build_loss_fn_passes_params_to_builtin_loss():
    """Params should be forwarded to the built-in loss class's constructor."""
    loss_fn, _ = build_loss_fn({
        'name': 'CrossEntropyLoss',
        'params': {
            'label_smoothing': 0.1
        },
    })

    assert loss_fn.label_smoothing == 0.1


def test_build_loss_fn_resolves_custom_loss_registered_under_scalr_nn_loss(
        monkeypatch):
    """A custom CustomLossBase subclass registered under scalr.nn.loss should be resolved by name."""

    class _DummyWeightedLoss(CustomLossBase):

        def __init__(self, gamma=1.0):
            super().__init__()
            self.gamma = gamma
            self.criterion = torch.nn.CrossEntropyLoss()

        @classmethod
        def get_default_params(cls):
            return {'gamma': 1.0}

    monkeypatch.setattr(loss_module,
                        '_DummyWeightedLoss',
                        _DummyWeightedLoss,
                        raising=False)

    loss_fn, loss_config = build_loss_fn({
        'name': '_DummyWeightedLoss',
        'params': {
            'gamma': 2.5
        },
    })

    assert isinstance(loss_fn, _DummyWeightedLoss)
    assert loss_fn.gamma == 2.5
    assert loss_config['params'] == {'gamma': 2.5}

    out = loss_fn(torch.randn(4, 3), torch.tensor([0, 1, 2, 1]))
    assert out.ndim == 0    # scalar loss


def test_build_loss_fn_custom_loss_fills_in_default_params(monkeypatch):
    """Unspecified params for a custom loss should fall back to get_default_params()."""

    class _DummyDefaultedLoss(CustomLossBase):

        def __init__(self, gamma=1.0):
            super().__init__()
            self.gamma = gamma
            self.criterion = torch.nn.CrossEntropyLoss()

        @classmethod
        def get_default_params(cls):
            return {'gamma': 1.0}

    monkeypatch.setattr(loss_module,
                        '_DummyDefaultedLoss',
                        _DummyDefaultedLoss,
                        raising=False)

    loss_fn, _ = build_loss_fn({'name': '_DummyDefaultedLoss'})

    assert loss_fn.gamma == 1.0


def test_build_loss_fn_raises_without_name():
    """A loss config without a `name` should raise, not silently no-op."""
    with pytest.raises(ValueError):
        build_loss_fn({})
