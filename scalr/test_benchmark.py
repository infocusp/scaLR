"""This is a test file for benchmark.py"""

from scalr.benchmark import capture_environment_metadata
from scalr.benchmark import peak_rss_mb
from scalr.benchmark import run_benchmark
from scalr.utils import generate_dummy_dge_anndata


def _toy_adata(n_donors, cell_replicate):
    return generate_dummy_dge_anndata(n_donors=n_donors,
                                      cell_type_list=['B_cell', 'T_cell', 'DC'],
                                      cell_replicate=cell_replicate,
                                      n_vars=15)


def test_run_benchmark_reports_quality_and_resource_metrics():
    """run_benchmark should produce one row per dataset with quality + cost metrics."""
    datasets = {
        'small': _toy_adata(n_donors=6, cell_replicate=5),
        'medium': _toy_adata(n_donors=6, cell_replicate=10),
    }

    df = run_benchmark(datasets,
                       labels_key='cell_type',
                       group_key='donor_id',
                       train_kwargs={
                           'hidden_layers': (8,),
                           'epochs': 1,
                           'batch_size': 32,
                           'device': 'cpu',
                       })

    assert list(df.index) == ['small', 'medium']
    for col in ('n_cells', 'n_genes', 'n_classes', 'train_seconds',
                'predict_seconds', 'cells_per_second_predict', 'peak_rss_mb',
                'macro_f1', 'balanced_accuracy'):
        assert col in df.columns
    assert (df['train_seconds'] > 0).all()
    assert (df['n_cells'] == [90, 180]).all()


def test_peak_rss_mb_is_positive():
    """Peak RSS should be a positive number of megabytes for a running process."""
    assert peak_rss_mb() > 0


def test_capture_environment_metadata_has_expected_keys():
    """Environment metadata should record versions and hardware availability."""
    meta = capture_environment_metadata()
    assert set(meta) == {
        'scalr_version', 'python_version', 'torch_version', 'cuda_available',
        'platform'
    }
