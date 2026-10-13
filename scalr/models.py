"""This file implements a local model registry, plus push/pull of model
artifacts to the Hugging Face Hub, for the scaLR 2.0 "model hub".

Hub access (`download`/`push`) requires the optional `huggingface_hub`
dependency (`pip install pyscaLR[hub]`) and is only imported lazily, so the
base install stays lightweight. Hub-hosted models are addressed by their HF
`repo_id` (`"<namespace>/<name>"`); local registry names are plain strings
with no `/`.
"""

import os
from os import path
import shutil
from typing import Optional

from scalr.utils import read_data

DEFAULT_REGISTRY_DIR = path.expanduser('~/.scalr/models')


def _require_hub():
    try:
        import huggingface_hub
    except ImportError as e:
        raise ImportError(
            'Hugging Face Hub access requires the optional "hub" extra. '
            'Install it with: pip install "pyscaLR[hub]"') from e
    return huggingface_hub


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


def _is_model_dir(model_dir: str) -> bool:
    return path.exists(path.join(model_dir, 'manifest.json')) or path.exists(
        path.join(model_dir, '_pointer.txt'))


def list_models(registry_dir: Optional[str] = None) -> list[str]:
    """List model names registered in the local registry.

    Hub-style names (`"<namespace>/<name>"`, e.g. from `download()`) are
    registered one directory level deeper than plain names, and are listed
    the same way, e.g. `"org/some-model"`.
    """
    reg_dir = _registry_dir(registry_dir)
    if not path.isdir(reg_dir):
        return []
    names = []
    for top in sorted(os.listdir(reg_dir)):
        top_dir = path.join(reg_dir, top)
        if not path.isdir(top_dir):
            continue
        if _is_model_dir(top_dir):
            names.append(top)
            continue
        for sub in sorted(os.listdir(top_dir)):
            sub_dir = path.join(top_dir, sub)
            if path.isdir(sub_dir) and _is_model_dir(sub_dir):
                names.append(f'{top}/{sub}')
    return names


def info(name: str, registry_dir: Optional[str] = None) -> dict:
    """Return a registered model's manifest dict."""
    return read_data(
        path.join(resolve_path(name, registry_dir), 'manifest.json'))


def download(
    repo_id: str,
    registry_dir: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """Download a scaLR model artifact from the Hugging Face Hub and
    register it locally.

    Args:
        repo_id: Hugging Face repo id to pull, e.g. `"infocusp/pbmc-v1"`.
        registry_dir: Registry root directory. Defaults to the
            `SCALR_MODEL_REGISTRY` environment variable, or `~/.scalr/models`.
        revision: Hub revision (branch/tag/commit) to pull. Defaults to the
            repo's default branch.
        token: Hugging Face auth token, for private repos. Defaults to the
            token `huggingface_hub` picks up from `HF_TOKEN`/`huggingface-cli
            login`.
        name: Local registry name to register the model under. Defaults to
            `repo_id` itself, so `scalr.load_model(repo_id)` resolves it
            without a separate `register()` call.

    Returns:
        The local path the model is registered/resolvable at.
    """
    hub = _require_hub()

    cache_dir = hub.snapshot_download(repo_id=repo_id,
                                      revision=revision,
                                      token=token,
                                      repo_type='model')

    if not path.exists(path.join(cache_dir, 'manifest.json')):
        raise ValueError(
            f'"{repo_id}" does not look like a scaLR model artifact '
            '(no manifest.json found in the downloaded repo).')

    return register(cache_dir, name or repo_id, registry_dir, copy=True)


def push(
    artifact_dir: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = 'Upload scaLR model artifact',
) -> str:
    """Upload a saved model artifact directory to the Hugging Face Hub.

    Args:
        artifact_dir: Path to a model artifact directory, as written by
            `AnnotationModel.save`.
        repo_id: Hugging Face repo id to publish to, e.g.
            `"infocusp/pbmc-v1"`. Created if it doesn't already exist.
        token: Hugging Face auth token. Defaults to the token
            `huggingface_hub` picks up from `HF_TOKEN`/`huggingface-cli
            login`.
        private: Whether to create the repo as private, if it doesn't
            already exist.
        commit_message: Commit message for the upload.

    Returns:
        The URL of the pushed model on the Hub.
    """
    if not path.exists(path.join(artifact_dir, 'manifest.json')):
        raise ValueError(
            f'"{artifact_dir}" does not look like a scaLR model artifact '
            '(no manifest.json found).')

    hub = _require_hub()

    api = hub.HfApi(token=token)
    api.create_repo(repo_id, repo_type='model', private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id,
                      folder_path=artifact_dir,
                      token=token,
                      commit_message=commit_message)

    return f'https://huggingface.co/{repo_id}'


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
