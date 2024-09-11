"""This is a test file for misc_utils.py"""

from copy import deepcopy

from scalr.utils import overwrite_default


def test_overwrite_default():
    """This funciton tests `overwrite_default()` function of misc_utils."""

    # User config key-values dictionary.
    user_config = {'a': 0, 'b': 1, 'd': 3}

    # Default config key-values dictionary.
    default_config = {'a': '5', 'b': 7, 'c': 2}

    # Getting updated default config using the overwrite function.
    updated_default_params = overwrite_default(
        user_config=user_config, default_config=deepcopy(default_config))

    # Checking whether the (key, values) not available in `user_config` are present in
    # `updated_default_params` and existing (key, values) are consistent with
    # `user_config` or not.
    for key in updated_default_params.keys():
        if key not in user_config:
            assert updated_default_params[key] == default_config[key]
        elif key in user_config:
            assert updated_default_params[key] == user_config[key]
