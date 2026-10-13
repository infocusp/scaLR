# Changelog

All notable changes to scaLR are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.2.0] - 2026-10-13

### Added
- Hugging Face Hub integration: `scalr.models.push`/`pull` and `scalr models push`/`pull` to publish and download model artifacts (`scalr/models.py`).
- Dataset drift detection: `scalr.detect_drift` / `scalr drift` to check whether a query dataset's distribution has shifted from a model's training data (`scalr/drift.py`).
- Explainability API: `scalr.explain` / `scalr explain` for gene-attribution explanations of model predictions (`scalr/explain.py`).
- Streaming/chunked prediction (`streaming=True`, `chunk_size=...`) in `scalr.annotate`/`AnnotationModel.predict` for bounded-memory inference on large datasets.
- Support for custom/user-defined loss functions via the `name`/`params` builder pattern (`scalr/nn/loss/_loss.py`).
- `scalr` CLI: `train`, `annotate`, `explain`, `evaluate`, `drift`, and `models {list,info,push,pull}` subcommands, plus a `[dev]` extra and Development setup docs in the README.
- `.gitignore` for local run artifacts (`.venv/`, `__pycache__/`, `example_data/`, `models/`, build output).

### Changed
- Feature scoring in `FeatureExtractionPipeline` parallelized across feature subsets.
- `DataIngestionPipeline`: `train_val_test` config checks consolidated into a single `_validate_train_val_test_config` method, run up front instead of being interleaved with split generation.

### Fixed
- Gene alignment when a dataset has duplicate gene names (`scalr/genes.py`, `scalr/feature/feature_subsetting.py`).
- Multi-worker config propagation during chunked preprocessing.
- `anndata` bumped past a broken 0.11.x chunked-read regression; fixed a `pydeseq2` `np.matrix` incompatibility in pseudobulk DGE (`scalr/analysis/dge_pseudobulk.py`).
- Disabled `AnnCollection` dtype harmonization, which was corrupting chunked `.h5ad` reads.

### Removed
- Dropped Python 3.10 support (`requires-python`, CI matrix, README). The `anndata>=0.12` floor needed for the chunked-read fix above requires Python >=3.11, so 3.10 installs could no longer resolve a compatible `anndata` version.

### Performance
- Lazy-import scanpy/SHAP-backed submodules (`scalr/analysis`, `scalr/feature/scoring`) to reduce base import time.

### Documentation
- `README.md` restructured into a concise entry point, with detailed content split out into linked guides: [docs/guide_simple_api.md](docs/guide_simple_api.md), [docs/guide_cli.md](docs/guide_cli.md), [docs/guide_pipeline.md](docs/guide_pipeline.md), and [docs/tutorials.md](docs/tutorials.md).
- Added [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, tests, and formatting instructions.
- Added Sphinx autodoc coverage for the entire simple-API layer (`docs/scalr.simple_api.rst`: `api`, `artifact`, `benchmark`, `calibration`, `cli`, `doublet`, `drift`, `explain`, `feature_stability`, `genes`, `hierarchy`, `leakage`, `metrics`, `model_card`, `models`, `refinement`, `result`, `validation`), which previously had no published documentation at all.
- Fixed broken/orphaned Sphinx toctree entries (`docs/modules.rst`, `docs/scalr.rst`) and resolved a duplicate-object-description warning on `scalr.result.PredictionResult`.
- Added the previously-missing SHAP analysis tutorial to the tutorials list ([docs/tutorials.md](docs/tutorials.md)).
- Fixed stale `anndata==0.10.9` pins in `dge.ipynb`, `gene_recall_curve.ipynb`, `normalization.ipynb`, and `batch_correction.ipynb` (incompatible with the `anndata>=0.12` floor above), and a stale feature-branch image URL in `scalr_pipeline.ipynb`/`scalr_pipeline_local_run.ipynb`.

## [1.1.0]

Baseline prior to this changelog's introduction. See `git log` for history before this point.
