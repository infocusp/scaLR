"""This is a test file for early_stopping.py"""

from copy import deepcopy

from scalr.nn.callbacks import EarlyStopping


def test_early_stopping():
    """This function tests early stopping of the model."""

    # Custom-defined validation loss to check early stopping.
    val_losses = [5, 2, 3, 2.1, 1.9, 3.0, 2.5, 2.0, 0.7, 0.4]
    patience = 3

    # The model should early stop at epoch 8 (val_loss=2.0) based on defined patience.
    expected_early_stop_epoch = 8

    # Creating objects for early stopping.
    early_stop = EarlyStopping(patience=patience)

    # Iterating over above val_losses to test epoch at which it is early stopping.
    observed_epochs = 1
    for val_loss in val_losses:
        if early_stop.__call__(val_loss=deepcopy(val_loss)):
            break
        observed_epochs += 1

    assert observed_epochs==expected_early_stop_epoch, f"There is some issue in early stopping."\
    f" Expected epochs({expected_early_stop_epoch}) != observed epoch({observed_epochs}). Please check!"
