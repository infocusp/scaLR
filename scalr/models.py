"""This file implements a local model registry: the starting point for the
scaLR 2.0 "model hub".

Per the v2 blueprint, the model hub starts as a local registry API
(`scalr.models.list()`, `.info()`, `.register()`) with remote hosted-model
downloads (`.download()`) left for a future release once actual hosting
infrastructure exists — `download()` here raises rather than silently
pretending to fetch something.
"""

import os
from os import path
import shutil
from typing import Optional

from scalr.utils import read_data

DEFAULT_REGISTRY_DIR = path.expanduser('~/.scalr/models')


def _registry_dir(registry_dir: Optional[str] = None) -> str:
    return registry_dir or os.environ.get('SCALR_MODEL_REGISTRY',
                                          DEFAULT_REGISTRY_DIR)


def _resolve(name: str, registry_dir: Optional[str] = None) -> str:
    target = path.join(_registry_dir(registry_dir), name)
    pointer_file = path.join(target, '_pointer.txt')
    if path.exists(pointer_file):
        with open(pointer_file) as fh:
            return fh.read().strip()
    return target


def register(
    artifact_dir: str,
    name: str,
    registry_dir: Optional[str] = None,
    copy: bool = True,
) -> str:
    """Register a saved model artifact directory under `name` in the local registry.

    Args:
        artifact_dir: Path to a model artifact directory, as written by
            `AnnotationModel.save`.
        name: Name to register the model under (used by `list`/`info`/`load_model`).
        registry_dir: Registry root directory. Defaults to the
            `SCALR_MODEL_REGISTRY` environment variable, or `~/.scalr/models`.
        copy: When True (default), copies the artifact into the registry.
            When False, records a pointer to `artifact_dir` instead of
            duplicating it on disk.

    Returns:
        The path the model is registered/resolvable at.
    """
    if not path.exists(path.join(artifact_dir, 'manifest.json')):
        raise ValueError(
            f'"{artifact_dir}" does not look like a scaLR model artifact '
            '(no manifest.json found).')

    reg_dir = _registry_dir(registry_dir)
    os.makedirs(reg_dir, exist_ok=True)
    target = path.join(reg_dir, name)

    if copy:
        if path.exists(target):
            shutil.rmtree(target)
        shutil.copytree(artifact_dir, target)
    else:
        os.makedirs(target, exist_ok=True)
        with open(path.join(target, '_pointer.txt'), 'w') as fh:
            fh.write(path.abspath(artifact_dir))

    return target


def list_models(registry_dir: Optional[str] = None) -> list[str]:
    """List model names registered in the local registry."""
    reg_dir = _registry_dir(registry_dir)
    if not path.isdir(reg_dir):
        return []
    return sorted(
        n for n in os.listdir(reg_dir) if path.isdir(path.join(reg_dir, n)))


def info(name: str, registry_dir: Optional[str] = None) -> dict:
    """Return a registered model's manifest dict."""
    return read_data(
        path.join(resolve_path(name, registry_dir), 'manifest.json'))


def download(name: str, registry_dir: Optional[str] = None) -> str:
    """Placeholder for future hosted-model downloads.

    No remote model hub is hosted yet; only locally `register`-ed models can
    be resolved. This raises rather than silently doing nothing, so calling
    code does not mistake a no-op for a successful download.
    """
    raise NotImplementedError(
        f'No hosted model hub is available yet; "{name}" must be registered '
        'locally first via scalr.models.register(artifact_dir, name).')


def resolve_path(name: str, registry_dir: Optional[str] = None) -> str:
    """Resolve a registered model name to its artifact directory path."""
    model_dir = _resolve(name, registry_dir)
    if not path.exists(path.join(model_dir, 'manifest.json')):
        raise FileNotFoundError(
            f'No model named "{name}" found in registry '
            f'"{_registry_dir(registry_dir)}". Registered models: '
            f'{list_models(registry_dir)}')
    return model_dir


# `list` matches the scaLR 2.0 blueprint's `scalr.models.list()` API; the
# builtin `list` is not otherwise used in this module.
list = list_models
