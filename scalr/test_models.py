"""This is a test file for models.py (local model registry)"""

import sys
from unittest.mock import MagicMock

import pytest

from scalr import models


def _fake_artifact(tmp_path, name='src_model'):
    artifact_dir = tmp_path / name
    artifact_dir.mkdir()
    (artifact_dir / 'manifest.json').write_text('{"labels": ["A", "B"]}')
    return str(artifact_dir)


def test_register_and_list_and_info_copy(tmp_path):
    """A registered model (copied) should show up in list() and info()."""
    artifact_dir = _fake_artifact(tmp_path)
    registry_dir = str(tmp_path / 'registry')

    target = models.register(artifact_dir,
                             'my_model',
                             registry_dir=registry_dir)

    assert models.list(registry_dir=registry_dir) == ['my_model']
    manifest = models.info('my_model', registry_dir=registry_dir)
    assert manifest['labels'] == ['A', 'B']
    # Copy mode: the registered artifact is independent of the source dir.
    assert target != artifact_dir


def test_register_pointer_mode_does_not_copy(tmp_path):
    """copy=False should register a pointer to the original directory, not duplicate files."""
    artifact_dir = _fake_artifact(tmp_path)
    registry_dir = str(tmp_path / 'registry')

    models.register(artifact_dir,
                    'ptr_model',
                    registry_dir=registry_dir,
                    copy=False)

    resolved = models.resolve_path('ptr_model', registry_dir=registry_dir)
    assert resolved == artifact_dir


def test_list_empty_registry_returns_empty_list(tmp_path):
    """An empty/nonexistent registry directory should list as empty, not raise."""
    registry_dir = str(tmp_path / 'does_not_exist')
    assert models.list(registry_dir=registry_dir) == []


def test_info_unknown_model_raises(tmp_path):
    """Requesting info for an unregistered model name should raise a clear error."""
    registry_dir = str(tmp_path / 'registry')
    with pytest.raises(FileNotFoundError):
        models.info('nonexistent', registry_dir=registry_dir)


def test_register_rejects_non_artifact_directory(tmp_path):
    """Registering a directory without manifest.json should raise."""
    not_an_artifact = tmp_path / 'random_dir'
    not_an_artifact.mkdir()
    with pytest.raises(ValueError):
        models.register(str(not_an_artifact),
                        'bad_model',
                        registry_dir=str(tmp_path / 'registry'))


def test_require_hub_missing_dependency_raises_helpful_error(monkeypatch):
    """Without the "hub" extra installed, hub access should fail with an actionable message."""
    monkeypatch.setitem(sys.modules, 'huggingface_hub', None)
    with pytest.raises(ImportError, match=r'pyscaLR\[hub\]'):
        models._require_hub()


def test_download_pulls_from_hub_and_registers_locally(monkeypatch, tmp_path):
    """download() should snapshot the hub repo and register it under repo_id by default."""
    snapshot_dir = _fake_artifact(tmp_path, name='snapshot')
    fake_hub = MagicMock()
    fake_hub.snapshot_download.return_value = snapshot_dir
    monkeypatch.setattr(models, '_require_hub', lambda: fake_hub)

    registry_dir = str(tmp_path / 'registry')
    target = models.download('org/some-model', registry_dir=registry_dir)

    fake_hub.snapshot_download.assert_called_once_with(
        repo_id='org/some-model',
        revision=None,
        token=None,
        repo_type='model',
    )
    assert models.list(registry_dir=registry_dir) == ['org/some-model']
    assert models.info('org/some-model', registry_dir=registry_dir) == {
        'labels': ['A', 'B']
    }
    assert target != snapshot_dir    # copied, not pointed at the cache dir.


def test_download_rejects_non_artifact_repo(monkeypatch, tmp_path):
    """A hub repo without a manifest.json should not be silently registered."""
    not_an_artifact = tmp_path / 'snapshot'
    not_an_artifact.mkdir()
    fake_hub = MagicMock()
    fake_hub.snapshot_download.return_value = str(not_an_artifact)
    monkeypatch.setattr(models, '_require_hub', lambda: fake_hub)

    with pytest.raises(ValueError):
        models.download('org/bad-model',
                        registry_dir=str(tmp_path / 'registry'))


def test_push_uploads_artifact_and_returns_url(monkeypatch, tmp_path):
    """push() should create the repo (if needed) and upload the artifact folder."""
    artifact_dir = _fake_artifact(tmp_path)
    fake_api = MagicMock()
    fake_hub = MagicMock()
    fake_hub.HfApi.return_value = fake_api
    monkeypatch.setattr(models, '_require_hub', lambda: fake_hub)

    url = models.push(artifact_dir, 'org/my-model', private=True)

    fake_hub.HfApi.assert_called_once_with(token=None)
    fake_api.create_repo.assert_called_once_with('org/my-model',
                                                 repo_type='model',
                                                 private=True,
                                                 exist_ok=True)
    fake_api.upload_folder.assert_called_once_with(
        repo_id='org/my-model',
        folder_path=artifact_dir,
        token=None,
        commit_message='Upload scaLR model artifact',
    )
    assert url == 'https://huggingface.co/org/my-model'


def test_push_rejects_non_artifact_directory(tmp_path):
    """Pushing a directory without manifest.json should raise before touching the hub."""
    not_an_artifact = tmp_path / 'random_dir'
    not_an_artifact.mkdir()
    with pytest.raises(ValueError):
        models.push(str(not_an_artifact), 'org/bad-model')
