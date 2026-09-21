"""This is a test file for models.py (local model registry)"""

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


def test_download_raises_not_implemented():
    """download() should clearly refuse rather than silently no-op, until a hub exists."""
    with pytest.raises(NotImplementedError):
        models.download('anything')
