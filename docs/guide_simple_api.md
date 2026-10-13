# Simple API guide

This guide covers the config-free `scalr.train`/`scalr.annotate`/`scalr.load_model` API in depth. For the 3-line hook, see the [Quickstart](https://github.com/infocusp/scaLR/blob/main/README.md#quickstart-annotate-cells-in-3-lines) in the main README; for the configuration-driven research pipeline, see the [Pipeline guide](guide_pipeline.md).

## Validate your data before training or annotating

```python
report = scalr.validate(adata, labels_key="cell_type", group_key="donor_id")
print(report)
```

`scalr.validate` (see [scalr/validation.py](https://github.com/infocusp/scaLR/blob/main/scalr/validation.py)) checks AnnData structure (empty/duplicate cells or genes), numerical issues (NaN/Inf, zero-count or unusually large-library-size cells), label/group columns, and reports the detected normalization state (raw counts, log1p-normalized, or scaled) — surfacing problems as an explicit report instead of a downstream stack trace.

## Gene alignment

Query data does not need to share the model's gene set or gene order. `model.predict`/`scalr.annotate` call [scalr/genes.py](https://github.com/infocusp/scaLR/blob/main/scalr/genes.py) internally to reorder/subset the query to the model's expected feature list, zero-filling any missing genes and reporting coverage (`result.metadata["gene_coverage"]`, `result.metadata["missing_features"]`). You can also run this alignment directly:

```python
aligned_adata, report = scalr.align_genes(adata, reference_features, min_feature_overlap=0.1)
print(report)
```

## Streaming / bounded-memory prediction

```python
result = model.predict("large_query.h5ad", streaming=True, chunk_size=20000)
# or, via scalr.annotate:
result = scalr.annotate(adata, model="models/pbmc_v1", streaming=True, chunk_size=20000)
```

Passing a file path to `model.predict`/`scalr.annotate` streams automatically; `streaming=True` on an in-memory `AnnData` also works. Gene alignment and scoring happen `chunk_size` cells at a time instead of materializing one aligned matrix for the whole dataset, bounding peak alignment/scoring memory regardless of total cell count. For out-of-core reads too (never loading the full raw dataset into memory), point it at a directory of chunked `.h5ad` files rather than a single large `.h5ad` file — see [scalr/artifact.py](https://github.com/infocusp/scaLR/blob/main/scalr/artifact.py).

## Self-contained model artifacts

`model.save("models/pbmc_v1")` writes a versioned, portable directory containing everything needed to reload and run the model — no external config required: `manifest.json`, `model.pt`, `model_config.json`, `label_mapping.json`, `features.json`, `preprocessing.json`, `calibration.json`, and `metrics.json`. `scalr.load_model("models/pbmc_v1")` reads it back into an `AnnotationModel` — see [scalr/artifact.py](https://github.com/infocusp/scaLR/blob/main/scalr/artifact.py).

## Leakage-safe evaluation & class imbalance

When `group_key` (e.g. `donor_id`) is passed to `scalr.train`, splitting keeps every group confined to a single split and checks for leakage across splits, warning if any group value still appears in more than one ([scalr/leakage.py](https://github.com/infocusp/scaLR/blob/main/scalr/leakage.py)). Training also reports class-size imbalance and uses inverse-frequency class weighting by default, and evaluates on the held-out test split with macro-F1, weighted-F1, balanced accuracy and a per-class report rather than relying on overall accuracy alone ([scalr/metrics.py](https://github.com/infocusp/scaLR/blob/main/scalr/metrics.py)).

## Hierarchical (coarse-to-fine) annotation

Pass a `taxonomy` mapping every fine-grained label to a broad label at training time, then request either level at inference time — without retraining:

```python
taxonomy = {"CD4_T": "T_cell", "CD8_T": "T_cell", "B_cell": "B_cell", "DC": "Myeloid"}
model = scalr.train(adata, labels_key="cell_type", group_key="donor_id", taxonomy=taxonomy)

fine_result = model.predict(adata, level="fine")     # default
broad_result = model.predict(adata, level="broad")   # aggregated via the taxonomy
```

See [scalr/hierarchy.py](https://github.com/infocusp/scaLR/blob/main/scalr/hierarchy.py); broad-class probabilities are the sum of the underlying fine-class probabilities.

## Cluster-aware refinement & doublet/mixed-cell screening

```python
result = scalr.annotate(
    adata,
    model="models/pbmc_v1",
    cluster_refinement="auto",  # or an adata.obs column name
    flag_doublets=True,
)
```

`cluster_refinement` relabels cells to their cluster's confidence-weighted majority class ([scalr/refinement.py](https://github.com/infocusp/scaLR/blob/main/scalr/refinement.py)) — the raw, unrefined labels are always kept in `result.metadata["raw_labels"]`, never silently discarded. `flag_doublets` sets `result.is_possible_doublet` for cells whose top-two class probabilities are both substantial and close together ([scalr/doublet.py](https://github.com/infocusp/scaLR/blob/main/scalr/doublet.py)) — a heuristic screening signal, not a validated doublet call.

## Explainability

```python
explanations = model.explain(adata, indices=[10, 20, 30], top_k=20)
print(explanations[0])

supporting, contradictory = model.explain_class("T_cell", top_k=50)
```

`model.explain` reports, per requested cell, the genes whose expression most supported or contradicted the model's prediction, via a Grad x Input gene-attribution score. `model.explain_class` does the same for a class in general, independent of any specific cell. This is a first-order sensitivity explanation, not a validated biological one — treat it as a screening/interpretation aid, like the doublet/refinement heuristics above, not ground truth ([scalr/explain.py](https://github.com/infocusp/scaLR/blob/main/scalr/explain.py)).

## Dataset drift detection

`is_unknown`/`is_possible_doublet` flag individual out-of-distribution cells, but can miss a whole cohort that has drifted (a different tissue, protocol, or batch) if most cells still score confidently. `scalr.train` records per-gene and per-cell (library size, detected-gene-count) statistics from the training split as a `drift_reference`; `scalr.detect_drift` compares a new dataset against it before you trust its predictions:

```python
report = scalr.detect_drift(query_adata, model="models/pbmc_v1")
print(report)
if report.is_drifted:
    ...  # treat predictions on this dataset with caution
```

A gene counts as drifted when its mean-expression shift exceeds `z_threshold` reference standard deviations (default 2); the dataset is flagged `is_drifted` when either too large a fraction of genes drift (`drift_fraction_threshold`, default 10%) or the library-size/detected-gene-count shift is extreme on its own (`severity_z_threshold`, default 5) — the latter catches e.g. a sequencing-depth change a per-gene test can miss. This is a univariate summary-statistics check, not a full distributional test — see [scalr/drift.py](https://github.com/infocusp/scaLR/blob/main/scalr/drift.py). Also available as `scalr drift --input query.h5ad --model models/pbmc_v1 [--fail-on-drift]` from the CLI, for use as a pre-flight/CI gate.

## Feature-selection stability

```python
report = scalr.compute_feature_stability(adata, labels_key="cell_type", top_k=50, n_runs=20)
print(report)
print(report.top_stable_features(min_runs=15))
```

Repeats feature selection over random subsamples and reports each gene's selection frequency plus the mean pairwise Jaccard similarity of the selected sets, to help distinguish a stable biomarker signal from a split-specific artifact ([scalr/feature_stability.py](https://github.com/infocusp/scaLR/blob/main/scalr/feature_stability.py)).

## Model cards, provenance & a model registry/hub

Every `model.save(...)` call also writes a `README.md` model card (training data, held-out metrics, calibration, known limitations, license) into the artifact directory ([scalr/model_card.py](https://github.com/infocusp/scaLR/blob/main/scalr/model_card.py)). Models can be registered and resolved by name via a local registry ([scalr/models.py](https://github.com/infocusp/scaLR/blob/main/scalr/models.py)):

```python
scalr.models.register("models/pbmc_v1", "human_pbmc")
scalr.models.list()          # -> ['human_pbmc']
scalr.models.info("human_pbmc")

model = scalr.load_model("human_pbmc")   # resolves the registered name
```

Models can also be published to, and pulled from, the [Hugging Face Hub](https://huggingface.co/models) — requires the optional `hub` extra (`pip install "pyscaLR[hub]"`):

```python
scalr.models.push("models/pbmc_v1", "your-org/human-pbmc")   # publish

model = scalr.load_model("your-org/human-pbmc")   # auto-downloads on first use, then cached locally
```

`scalr.load_model` tries an on-disk path, then the local registry, and only falls back to a Hub download for `"<namespace>/<name>"`-shaped names it can't resolve locally; once downloaded, a model is registered under that same name so later loads are local and offline. Use `scalr.models.download(repo_id, revision=..., token=...)` directly for a specific revision or a private repo.

## Benchmark suite

```python
df = scalr.run_benchmark(
    {"small": small_adata, "medium": medium_adata},
    labels_key="cell_type",
    group_key="donor_id",
)
```

Reports scientific quality (macro-F1, balanced accuracy) alongside computational cost (wall time, cells/sec, peak RSS) per dataset in one table, so quality and resource cost are always reported together rather than a one-off number ([scalr/benchmark.py](https://github.com/infocusp/scaLR/blob/main/scalr/benchmark.py)).
