# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

scaLR (Single-cell analysis using Low Resource) is a Python library/pipeline for scRNA-seq cell-type classification, feature selection, and biomarker analysis, designed to work on large datasets with limited compute. It has two layers:

1. **Simple library API** (`scalr.train`, `scalr.annotate`, `scalr.load_model`, `scalr.validate`, `scalr.align_genes`) — an inference-first API added on top of the engine for quick, config-free use. See `scalr/api.py`, `scalr/artifact.py`, `scalr/result.py`.
2. **Configuration-driven pipeline** (`pipeline.py` + a YAML config) — the original, full-featured research pipeline: data ingestion/splitting/preprocessing → feature extraction → model training → evaluation & downstream analysis. This remains the primary way to run large, customized experiments.

Both layers share the same underlying engine in `scalr/` (models, dataloaders, trainers, losses, splitters, preprocessors, feature scorers/selectors, analysis modules).

## Commands

Install (editable, for development):
```bash
pip install -r requirements.txt
pip install -e .
```
CPU is the default `torch` install; for CUDA, install a CUDA-specific `torch` build per the [PyTorch instructions](https://pytorch.org/get-started/locally/) before or after. `requirements.txt` pins `pandas<3` and `numpy<2` deliberately — unpinned, a fresh resolve can pull `pandas` 3.x / `numpy` 2.x, which break h5ad writing (`anndata`) and `shap` respectively; don't remove those pins without re-verifying the full test suite from a clean install.

`pip install -e .` also registers the `scalr` console script (`scalr validate|train|annotate|evaluate|models ...`, see `scalr/cli.py`), a thin wrapper around the simple API below.

Run the full config-driven pipeline:
```bash
python pipeline.py --config /path/to/config.yaml -l -m
```
(`-l`/`--log` enables experiment logging, `-m`/`--memoryprofiler` enables memory profiling.)

Tests (pytest, no separate test directory — test files live next to the code as `test_*.py`):
```bash
pytest                          # run everything
pytest -v -s                    # verbose, as CI does
pytest scalr/test_api.py        # a single test file
pytest scalr/test_api.py::test_train_returns_annotation_model_with_metrics   # a single test
```

Formatting/import order (both enforced in CI, config in `.style.yapf` / `.isort.cfg`, both based on the `google` style, 80-col):
```bash
yapf -ir --style .style.yapf .
isort . --settings-path .isort.cfg
```
`pre-commit` is configured (`.pre-commit-config.yaml`) to run both automatically.

Docs (Sphinx, published to GitHub Pages on push to `main`):
```bash
sphinx-build docs _build
```

## Architecture

### Config-driven pipeline (`scalr/*_pipeline.py`, driven by `pipeline.py`)

`pipeline.py` reads one YAML config and runs whichever top-level sections are present (sections can be omitted/commented out to skip stages):

1. **`data`** → `DataIngestionPipeline` (`data_ingestion_pipeline.py`): splits data into train/val/test (via `scalr/data/split/`), applies preprocessing (`scalr/data/preprocess/`), generates label mappings.
2. **`feature_selection`** → `FeatureExtractionPipeline` (`feature_extraction_pipeline.py`): trains on feature subsets batch-wise, scores features (`scalr/feature/scoring/`: linear-weight or SHAP-based), and selects top-k features (`scalr/feature/selector/`).
3. **`final_model`**/training → `ModelTrainingPipeline` (`model_training_pipeline.py`): builds model/optimizer/loss/callbacks and trains via a `Trainer` (`scalr/nn/trainer/`).
4. **`analysis`** → `EvalAndAnalysisPipeline` (`eval_and_analysis_pipeline.py`): evaluation metrics/classification report, gene/biomarker analysis, and pluggable downstream analyses (`scalr/analysis/`: DGE via pseudobulk or linear mixed models, ROC-AUC, gene recall curves, heatmaps).

Each stage returns an updated config dict so the full resolved config (including defaults) gets written back out — this is how `config/README.md` params map to code.

### The `name`/`params` builder pattern

Almost every pluggable component (models, dataloaders, loss functions, splitters, preprocessors, feature scorers/selectors, callbacks, analysers) is instantiated from a config dict of the form `{name: <ClassName>, params: {...}}` via `scalr.utils.build_object` (`scalr/utils/misc_utils.py`), which looks up `<ClassName>` in the given module, merges `params` over the class's `get_default_params()`, and constructs it. Each such subpackage exposes a `build_*` convenience function next to its base class (e.g. `build_model` in `scalr/nn/model/_model.py`, `build_preprocessor` in `scalr/data/preprocess/_preprocess.py`, `build_splitter`, `build_analyser`, `build_dataloader`, `build_loss_fn`). When adding a new implementation of any of these, follow the existing sibling classes' pattern: subclass the module's base class, implement its abstract methods, and implement `get_default_params()`.

### Data handling

- Input is `.h5ad` (AnnData) or a directory of chunked `.h5ad` files, read via `scalr.utils.read_data`/`write_data` (`scalr/utils/file_utils.py`), which returns an `AnnCollection` for chunked directories to avoid loading everything into memory.
- `adata.X` is expected to already be normalized (`log1p`, range 0–10) for the pipeline convention; `adata.obs` holds cell metadata including the classification target column; `adata.var` holds gene names.
- Chunkwise writing/transformation (`write_chunkwise_data`) is the mechanism used throughout to keep peak memory bounded regardless of dataset size — this is central to the "low resource" design goal, so avoid introducing full-dataset `.toarray()`/dense materialization in new code.
- `scalr.data.split` splitters (`StratifiedSplitter`, `GroupSplitter`, `StratifiedGroupSplitter`) matter for scientific validity: `GroupSplitter`/`StratifiedGroupSplitter` keep a `stratify` column (e.g. `donor_id`) confined to a single split to prevent train/test leakage — prefer these over plain stratified splitting whenever a donor/patient/sample identifier is available.

### Simple API layer (`scalr/api.py`, `scalr/artifact.py`, `scalr/result.py`, plus one module per capability)

This is a newer, self-contained layer built on top of the same engine (`scalr.nn.model`, etc.) but bypassing the config/pipeline machinery for simple in-memory workflows. Each capability lives in its own top-level module and is wired into `AnnotationModel`/`api.py` as an optional parameter, not a separate code path:

- `scalr.train(adata, labels_key, group_key=...)` does leakage-safe splitting (`scalr/leakage.py`), class-weighted training, temperature-scaling calibration (`scalr/calibration.py`), and macro/weighted-F1 + balanced-accuracy evaluation (`scalr/metrics.py`), returning an `AnnotationModel`. Pass `taxonomy={fine_label: broad_label, ...}` to enable hierarchical annotation.
- `AnnotationModel.save()`/`scalr.load_model()` (`scalr/artifact.py`) persist/restore a **self-contained model artifact directory** (`manifest.json`, `model.pt`, `model_config.json`, `label_mapping.json`, `features.json`, `preprocessing.json`, `calibration.json`, `metrics.json`, `taxonomy.json` if set, and an auto-generated `README.md` model card via `scalr/model_card.py`) — no external config needed to reload and run inference. `scalr.load_model` also resolves a bare name via the local model registry (`scalr/models.py`, `scalr.models.list/info/register`; `download()` intentionally raises — no hosted hub yet).
- `scalr.annotate(adata, model=...)` / `AnnotationModel.predict()` validate the input (`scalr/validation.py`), align its genes to the model's expected feature list (`scalr/genes.py`, zero-filling missing genes), run inference, and return a typed `PredictionResult` (`scalr/result.py`) with calibrated `confidence`, `entropy`, `margin`, and an `is_unknown` open-set abstention flag — always aligned to `adata.obs_names` (`PredictionResult.write_to_adata` refuses to write if names don't match, to prevent silently misaligned predictions). Optional `predict()`/`annotate()` params layer on: `level="broad"` (aggregate via taxonomy, `scalr/hierarchy.py`), `cluster_refinement="auto"|<obs_key>` (confidence-weighted per-cluster majority vote, `scalr/refinement.py` — always preserves the pre-refinement labels in `result.metadata["raw_labels"]`), and `flag_doublets=True` (sets `result.is_possible_doublet`, a heuristic screening signal from `scalr/doublet.py`, not a validated doublet call).
- `scalr.compute_feature_stability(adata, labels_key, ...)` (`scalr/feature_stability.py`) repeats feature selection over resampled subsets (via `sklearn.LogisticRegression`, independent of the DNN training loop) and reports per-gene selection frequency and Jaccard stability — for judging whether a selected biomarker set is a stable signal or a split-specific artifact.
- `scalr.run_benchmark(datasets, labels_key, ...)` (`scalr/benchmark.py`) runs train+predict per dataset and reports both quality (macro-F1, balanced accuracy) and cost (wall time, cells/sec, peak RSS via `resource.getrusage`) together, plus environment metadata — never scale one dataset's numbers without the other.
- `scalr/cli.py` exposes `validate`/`train`/`annotate`/`evaluate`/`models list`/`models info` as the `scalr` console script (registered in `pyproject.toml` under `[project.scripts]`); it's a thin argparse wrapper around this same API, not a separate implementation.

When extending this layer, keep it usable without any YAML config — pass all such new functionality through function parameters. This layer works with in-memory `AnnData` directly rather than the chunked/config-driven flow used by the pipeline, so it's not currently intended for atlas-scale/out-of-core datasets.

### Logging

Two logger classes in `scalr/utils/logger.py`: `FlowLogger` prints high-level stage progress to stdout (use this for anything user-facing/visible by default). `EventLogger` writes detailed logs to a file and is **silent by default** (`NullHandler`) unless a `filepath` or `stdout=True` is given — don't use it where output should be visible without extra configuration.
