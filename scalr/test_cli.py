"""This is a test file for cli.py"""

from scalr.cli import main
from scalr.utils import generate_dummy_dge_anndata
from scalr.utils import write_data


def _write_toy_h5ad(path_):
    adata = generate_dummy_dge_anndata(
        n_donors=8,
        cell_type_list=['B_cell', 'T_cell', 'DC'],
        cell_replicate=10,
        n_vars=20)
    write_data(adata, str(path_))
    return adata


def test_cli_validate_returns_zero_for_usable_data(tmp_path, capsys):
    """`scalr validate` should print a report and exit 0 for usable data."""
    input_path = tmp_path / 'data.h5ad'
    _write_toy_h5ad(input_path)

    code = main([
        'validate', '--input',
        str(input_path), '--labels-key', 'cell_type', '--group-key', 'donor_id'
    ])

    captured = capsys.readouterr()
    assert code == 0
    assert 'scaLR validation' in captured.out


def test_cli_validate_returns_nonzero_for_missing_labels_key(tmp_path, capsys):
    """`scalr validate` should exit non-zero when labels_key is missing."""
    input_path = tmp_path / 'data.h5ad'
    _write_toy_h5ad(input_path)

    code = main([
        'validate', '--input',
        str(input_path), '--labels-key', 'does_not_exist'
    ])
    assert code == 1


def test_cli_train_then_annotate_roundtrip(tmp_path, capsys):
    """`scalr train` then `scalr annotate` should produce an annotated h5ad file."""
    input_path = tmp_path / 'data.h5ad'
    _write_toy_h5ad(input_path)
    model_dir = tmp_path / 'model'
    output_path = tmp_path / 'annotated.h5ad'

    train_code = main([
        'train', '--input',
        str(input_path), '--labels-key', 'cell_type', '--group-key', 'donor_id',
        '--output',
        str(model_dir), '--epochs', '2', '--device', 'cpu', '--quiet'
    ])
    assert train_code == 0
    assert (model_dir / 'manifest.json').exists()
    assert (model_dir / 'README.md').exists()

    annotate_code = main([
        'annotate', '--input',
        str(input_path), '--model',
        str(model_dir), '--output',
        str(output_path), '--device', 'cpu'
    ])
    captured = capsys.readouterr()
    assert annotate_code == 0
    assert output_path.exists()
    assert 'Annotated' in captured.out


def test_cli_models_list_reports_empty_registry(tmp_path, monkeypatch):
    """`scalr models list` should not crash against an empty/default registry."""
    monkeypatch.setenv('SCALR_MODEL_REGISTRY', str(tmp_path / 'empty_registry'))
    code = main(['models', 'list'])
    assert code == 0
