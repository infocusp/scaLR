# Changelog

All notable changes to scaLR are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

## [1.1.0]

Baseline prior to this changelog's introduction. See `git log` for history before this point.
