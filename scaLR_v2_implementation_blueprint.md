# scaLR 2.0 — Comprehensive Implementation Blueprint

**Date:** 2026-09-21  
**Repository:** https://github.com/infocusp/scaLR  
**Paper:** https://doi.org/10.1093/bib/bbaf243  
**Scope:** API redesign, scientific reliability, scalability, reproducibility, packaging, model distribution, visualization, benchmarking, and ecosystem integration.

---

## 0. Executive summary

scaLR's strongest technical identity is **low-resource, large-scale single-cell analysis using DNN-based feature selection, classification, and biomarker analysis**. The published work focuses on reducing computational requirements while maintaining competitive classification performance at large cell counts. citeturn794974search2

The current repository is an end-to-end, configuration-driven pipeline. It expects `.h5ad` input, documents normalized `adata.X`, uses a YAML configuration for pipeline execution, and currently exposes the main pipeline stages rather than a concise inference-first API. citeturn799194search2

### Recommended v2 positioning

Do **not** turn scaLR into another generic deep-learning framework. Instead evolve it from:

> **pipeline → library + pipeline**

and position v2 around four properties:

1. **Easy:** `scalr.annotate(adata, model="human_pbmc")`
2. **Reliable:** validation, calibration, abstention, leakage-safe evaluation, rare-cell metrics, open-set detection.
3. **Scalable:** sparse processing, bounded-memory inference, chunking/streaming, CPU/GPU auto-selection.
4. **Interpretable/reproducible:** feature stability, explanations, model cards, self-contained artifacts, run metadata.

The result should be useful both to a beginner who wants cell-type labels and to an expert running million- or multi-million-cell analyses.

---

# 1. Current-state assessment

The public repository currently contains `scalr/`, `pipeline.py`, configuration, documentation and tutorials. The README describes `.h5ad` as the current input format and expects normalized expression values in `adata.X`; the example configuration uses a `GroupSplitter` stratified by donor. citeturn799194search2

The repository also has open enhancement items around configuration validation, unified logging, abstract base classes, more tests/CI, and Conda installation. These should be folded into the v2 foundation rather than treated as isolated cleanup tasks.

### Current architectural strengths

- Existing separation between ingestion, feature extraction, model training, and evaluation/analysis.
- Explicit chunk-size/resource configuration.
- Group-based data splitting is already demonstrated in the example configuration. citeturn799194search2
- Feature-selection and biomarker-analysis capabilities align with the paper's contribution. citeturn794974search2
- Existing CLI/pipeline can remain as the advanced/research workflow.

### Current barriers to adoption

- Pipeline-first workflow rather than inference-first library API.
- Users need to understand YAML/configuration before getting a simple prediction.
- `.h5ad` is the primary documented input path and preprocessing expectations are strict. citeturn799194search2
- Installation is more constrained than ideal for CPU/GPU/HPC environments.
- No clear first-class model artifact/model-hub contract.
- No first-class confidence/abstention/open-set workflow.
- No explicit contract for gene alignment and preprocessing transfer at inference time.
- Limited guardrails against leakage, rare-cell failure, mixed/doublet cells, and distribution shift.
- Benchmarking can be strengthened to cover calibration, unknown-cell detection, scaling curves, and reproducibility.

---

# 2. Product architecture for scaLR 2.0

Use two user layers over the same internal engine.

```text
                       scaLR 2.0
                           |
          +----------------+----------------+
          |                                 |
       Simple API                       Advanced API
          |                                 |
     annotate()                         YAML pipeline
     train()                            Pipeline classes
     load_model()                       Custom components
     explain()                          Research configs
          |                                 |
          +----------------+----------------+
                           |
                    Shared core engine
                           |
        +------------------+------------------+
        |                  |                  |
   Data/Preprocess      Model/Inference    Evaluation
        |                  |                  |
   AnnData/genes      DNN/selection       metrics
   normalization      uncertainty         benchmarks
   validation         explanation         reporting
```

### Design principle

Do not rewrite the current system in one large refactor. Introduce the v2 public API as a thin, stable layer over the current components, then refactor internals incrementally.

---

# 3. Public API

## 3.1 One-line annotation

```python
import scalr

result = scalr.annotate(
    adata,
    model="human_pbmc",
    device="auto",
    preprocess="auto",
)

adata.obs["scalr_cell_type"] = result.labels
adata.obs["scalr_confidence"] = result.confidence
adata.obs["scalr_unknown"] = result.is_unknown
```

## 3.2 Training

```python
model = scalr.train(
    adata,
    labels_key="cell_type",
    group_key="donor_id",
    device="auto",
)

model.save("models/pbmc_v1")
```

## 3.3 Loading and inference

```python
model = scalr.load_model("models/pbmc_v1")
result = model.predict(adata)
```

## 3.4 Explanation

```python
explanation = model.explain(
    adata,
    indices=[10, 20, 30],
    top_k=20,
)
```

## 3.5 Validation

```python
report = scalr.validate(
    adata,
    model="human_pbmc",
)
```

---

# 4. Prediction result contract

Return a typed result object rather than a bare array.

```python
result.labels
result.probabilities
result.top_k
result.confidence
result.entropy
result.is_unknown
result.ood_score
result.margin
result.explanations
result.metadata
```

Recommended conceptual structure:

```text
PredictionResult
├── labels
├── probabilities
├── confidence
├── entropy
├── top_k
├── margin
├── is_unknown
├── ood_score
├── explanations
└── metadata
```

The result must preserve row/cell identity so predictions cannot silently become misaligned with `adata.obs_names`.

---

# 5. AnnData-first integration

AnnData should become the first-class Python interface.

### Standard output fields

```python
adata.obs["scalr_pred"]
adata.obs["scalr_confidence"]
adata.obs["scalr_unknown"]
```

Optional:

```python
adata.obsm["scalr_probabilities"]
adata.obsm["scalr_embeddings"]
adata.uns["scalr"]
```

`adata.uns["scalr"]` should contain enough metadata to reconstruct the model/preprocessing contract used for that annotation.

### Preserve compatibility

Keep the existing `.h5ad` workflow. Add support rather than breaking it.

Later adapters can support:

- Zarr
- Seurat via conversion
- SingleCellExperiment/Bioconductor via a documented bridge
- streaming readers

---

# 6. Input validation subsystem

Create a formal validation layer instead of letting downstream PyTorch, NumPy or scikit-learn exceptions reveal implementation details.

## 6.1 Validation checks

### Data structure

- AnnData object or supported path.
- Non-empty observations and variables.
- Unique cell IDs.
- Unique gene IDs.
- Valid matrix dimensions.
- Supported sparse formats.

### Numerical checks

- NaN/Inf detection.
- Negative values handled explicitly.
- Zero-count cells reported.
- Extremely large library sizes reported.
- Optional mitochondrial/QC metadata checks.

### Metadata checks

During training:

- label column exists.
- grouping column exists when required.
- missing labels are reported.
- class sizes are reported.

### Model compatibility

- species.
- expected gene identifier type.
- expected preprocessing.
- required feature coverage.
- label compatibility.

## 6.2 User-facing report

```text
scaLR validation
----------------
ERROR
  None

WARNING
  179/5,000 model genes are missing from query.

INFO
  Input: 125,431 cells × 18,642 genes
  Sparse: yes
  Detected expression state: log1p-normalized
  Model: human_pbmc_v1
  Gene coverage: 96.4%

Status: usable with warning
```

## 6.3 No silent transformations

`preprocess="auto"` should detect and explain the state.

`preprocess="force"` may apply an explicit documented transformation.

Ambiguous inputs should trigger an informative warning/error rather than guessing.

---

# 7. Preprocessing contract

Separate **fit-time preprocessing** from **inference-time application**.

At training time, store:

- normalization method.
- layer used.
- scaling parameters where applicable.
- selected genes and ordering.
- preprocessing version.
- optional QC thresholds.

At inference time, apply exactly the stored contract.

### Important rule

Never calculate training-dependent statistics from the query set when the model was trained on a reference dataset.

This prevents data leakage and inconsistent predictions.

---

# 8. Gene identifier normalization and alignment

Gene alignment should be an explicit subsystem.

Support, where appropriate:

- gene symbols.
- Ensembl IDs.
- species-specific normalization.
- case normalization.
- documented mapping tables.

Example:

```python
aligned = scalr.align_genes(
    query,
    reference_model,
    species="human",
)
```

Output:

```text
Reference features: 5,000
Query genes:        18,642
Matched features:    4,821 (96.4%)
Missing features:      179
Extra query genes:  13,821

Status: usable with warning
```

Add configurable `min_feature_overlap` and fail early when coverage is inadequate.

---

# 9. Reference / query model

Make the distinction explicit.

## Training mode

```text
Labeled reference
      |
      v
Feature selection
      |
      v
DNN training
      |
      v
Calibration / evaluation
      |
      v
Versioned model artifact
```

## Annotation mode

```text
Versioned reference model
           +
        Query h5ad
           |
           v
Gene alignment
           |
           v
Preprocessing contract
           |
           v
Batch/chunk inference
           |
           v
Labels + confidence + uncertainty
```

The second path should not retrain the model unless explicitly requested.

---

# 10. Confidence calibration and uncertainty

A softmax maximum should not automatically be described as a biologically calibrated confidence value.

Implement:

- temperature scaling first.
- calibration dataset/holdout handling.
- Expected Calibration Error.
- Brier score.
- reliability diagrams.

Return:

- top-1 label.
- calibrated confidence.
- top-k probabilities.
- predictive entropy.
- top-1/top-2 margin.
- optional OOD score.

Example:

```text
Prediction: T cell
Calibrated confidence: 0.78
Entropy: 0.82
Top-2 margin: 0.11
Status: UNCERTAIN
```

---

# 11. Open-set / unknown-cell detection

Closed-set classification forces every cell into a known class. scaLR 2.0 should support abstention.

```python
result = model.predict(
    adata,
    open_set=True,
)
```

Possible output:

```text
label: UNKNOWN
confidence: 0.38
ood_score: high
```

Initial implementation:

- configurable confidence threshold.
- entropy threshold.
- top-1/top-2 margin.

Later evaluation/experimentation:

- energy score.
- Mahalanobis distance.
- ensemble disagreement.

Benchmark against held-out novel cell types.

---

# 12. Rare-cell and class-imbalance support

Do not rely on overall accuracy.

Add options for:

- weighted cross entropy.
- class-balanced sampling.
- focal loss as an optional method.
- minimum class-size warnings.

Report:

- macro-F1.
- weighted-F1.
- balanced accuracy.
- per-class precision/recall/F1.
- rare-class recall.
- confusion matrix.

Example warning:

```text
Class imbalance detected

T cell: 500,000
B cell: 200,000
DC:         800

Recommendation: enable class-balanced sampling.
```

---

# 13. Leakage-safe evaluation

The current examples already demonstrate donor-grouped splitting; make this a first-class safety feature rather than only a configuration option. citeturn799194search2

Support:

```python
scalr.train(
    adata,
    labels_key="cell_type",
    group_key="donor_id",
    split_strategy="group",
)
```

Detect likely leakage when the same donor/patient/sample/batch appears across train/test.

Warnings should identify the suspected grouping variable and recommend grouped splitting.

### Default philosophy

For reference datasets, prefer group-safe splits when donor/patient/sample identifiers are available.

---

# 14. Hierarchical cell-type annotation

Add optional coarse-to-fine classification.

```text
Immune
├── Lymphoid
│   ├── T cell
│   │   ├── CD4
│   │   └── CD8
│   ├── B cell
│   └── NK
└── Myeloid
    ├── Monocyte
    └── DC
```

API:

```python
result = model.predict(
    adata,
    level="broad",
)

result = model.predict(
    adata,
    level="subtype",
)
```

This allows scaLR to represent the difference between evidence for a broad type and evidence for a specific subtype.

---

# 15. Cluster-aware refinement

Provide optional post-processing using neighborhood/cluster information.

```python
result = model.predict(
    adata,
    cluster_refinement="auto",
)
```

Always preserve both:

```text
raw_prediction
refined_prediction
```

Never overwrite the raw model output without explicit user request.

---

# 16. Doublet / mixed-cell handling

A classifier should not silently force an obviously mixed cell into a single lineage.

Add an optional mixed-cell signal using:

- high predictive entropy.
- conflicting marker programs.
- top-two class margin.
- optional integration with a dedicated doublet detector.

Possible output:

```text
Prediction: T cell
Status: POSSIBLE_DOUBLET
T cell: 0.48
B cell: 0.43
```

The doublet detection feature should be labeled as a screening signal, not a definitive biological diagnosis, unless validated against a specific detector/dataset.

---

# 17. Distribution-shift robustness

Evaluate reference/query differences in:

- donor.
- tissue.
- disease state.
- sequencing technology.
- sequencing depth.
- batch.
- species, when supported.

Expose a compatibility report:

```text
Reference: human PBMC / 10x
Query:     human PBMC / Smart-seq2

Gene coverage: 92.1%
Distribution shift: detected
Recommendation: validate on domain-matched reference before production use.
```

Do not claim universal cross-domain robustness without benchmark evidence.

---

# 18. Feature-selection stability

Feature selection is a central scientific feature of scaLR. The paper describes subset-based feature analysis and final feature selection as part of the workflow. citeturn794974search2

Add stability analysis:

```text
Run 1 → top 500
Run 2 → top 500
Run 3 → top 500
...
Run 100 → top 500
```

Compute:

- selection frequency.
- rank stability.
- Jaccard similarity.
- class-specific stability.

Example:

```text
Gene    Selected runs / 100
CD3D        98
CD3E        97
TRBC2       91
IL7R        88
```

This makes feature selection more defensible as a biomarker-discovery aid.

---

# 19. Separate predictive features from biological biomarkers

Do not equate model importance with biological causality or biomarker specificity.

Expose separate reports:

```text
1. Predictive model features
2. Class-specific features
3. Differentially expressed genes
4. Known/reference markers
5. Feature-selection stability
```

Naming should make this distinction explicit.

---

# 20. Explainability API

Turn the paper's feature-importance capability into an accessible user feature. citeturn794974search2

```python
explanation = model.explain(
    adata,
    indices=[10, 20, 30],
    top_k=20,
)
```

For each cell:

```text
cell_10392
Prediction: T cell
Confidence: 0.94

Supporting genes
CD3D  +0.42
CD3E  +0.37
TRBC2 +0.29
IL7R  +0.21

Contradictory signals
MS4A1 -0.02
```

Also provide:

```python
model.explain_class("T_cell", top_k=50)
```

---

# 21. Marker sanity checks

Optionally compare predictions to expected marker programs.

Example:

```text
Predicted: CD4 T cell

Supporting
CD3D   ✓
CD3E   ✓
TRBC1  ✓
IL7R   ✓

Conflicting
MS4A1  ✗
NKG7   low
```

This is an explanation/sanity check, not a substitute for biological validation.

---

# 22. Batch correction safety

If batch correction is supported, provide diagnostics instead of exposing it as a silent switch.

Report before/after measures for:

- batch mixing.
- cell-type conservation.
- neighborhood structure.
- cluster preservation.

The goal is not merely to reduce batch separation; it is to reduce technical separation while preserving biological structure.

---

# 23. Model artifact format

Every model should be self-contained and versioned.

Recommended format:

```text
model/
├── manifest.json
├── model.safetensors
├── model_config.json
├── label_mapping.json
├── features.json
├── preprocessing.json
├── calibration.json
├── metrics.json
└── README.md
```

### `manifest.json`

```json
{
  "format_version": 1,
  "scalr_version": "2.x",
  "model_name": "human_pbmc",
  "model_version": "1.0.0",
  "task": "cell_type_annotation",
  "species": "human",
  "feature_count": 5000,
  "labels": ["B_cell", "T_cell", "NK_cell"],
  "normalization": "log1p_cp10k",
  "calibration": "temperature_scaling",
  "random_seed": 42
}
```

Prefer a portable state/weights representation such as `safetensors` where compatible rather than relying on arbitrary Python object deserialization.

---

# 24. Model provenance and model cards

Every published/pretrained model should have a model card containing:

- training/reference datasets.
- species.
- tissue.
- technology.
- number of cells.
- donor count.
- cell types.
- normalization assumptions.
- input gene coverage.
- performance metrics.
- unknown-cell behavior.
- known limitations.
- model version.
- code version.
- model/data license information.

Separate:

```text
source-code license
model license
training-data license
paper license
```

The current repository identifies the project as GPL-3.0, while the journal article has its own publication license terms; future model distribution should make those boundaries explicit. citeturn799194search2turn794974search2

---

# 25. Model registry / model hub

Start with a local registry API:

```python
scalr.models.list()
scalr.models.info("human_pbmc")
scalr.models.download("human_pbmc")
```

Possible model families:

```text
Human
├── PBMC
├── Lung
├── Brain
├── Liver
└── Pan-tissue

Mouse
├── Brain
├── Immune
└── Development

Disease
├── Cancer
├── Autoimmune
└── Other validated cohorts
```

Do not populate models merely for completeness. Every model should have a reproducible provenance record and benchmark/model card.

---

# 26. Automatic model selection

Provide:

```python
result = scalr.auto_annotate(adata)
```

The selector can inspect:

- species metadata.
- tissue metadata where available.
- gene overlap.
- supported labels.
- model provenance.
- compatibility constraints.

It should explain the selected model:

```text
Selected: human_lung_v2

Reason:
✓ human gene identifiers
✓ high feature overlap
✓ lung-associated model
✓ compatible normalization
```

Do not make hidden model choices. The selected model and reason must be recorded.

---

# 27. Device / resource auto-selection

Defaults:

```python
device="auto"
batch_size="auto"
num_workers="auto"
```

Inspect:

- CUDA availability.
- GPU VRAM.
- system RAM.
- number of cells.
- number of selected genes.
- sparse/dense input.

Log the chosen resources before execution.

Allow full override for reproducibility:

```python
device="cpu"
batch_size=4096
num_workers=4
```

---

# 28. Sparse and bounded-memory inference

Preserve sparse matrices until conversion to dense tensors is unavoidable.

Avoid unnecessary `.toarray()` / `.A` / dense materialization.

Provide:

```python
model.predict(
    "large_dataset.h5ad",
    streaming=True,
    chunk_size=20000,
)
```

A bounded-memory test must prove that peak RAM does not grow linearly with total cell count for the intended streaming workflow.

---

# 29. Out-of-core and Zarr/cloud roadmap

Later support:

- H5AD chunked reading.
- Zarr.
- object storage where supported.
- remote references.

Example future API:

```python
model.predict(
    "s3://bucket/atlas.zarr",
    streaming=True,
)
```

Do not advertise cloud execution before it is tested end-to-end.

---

# 30. Lightweight and low-resource model variants

Make resource efficiency an explicit product feature.

Model families could include:

```text
human_pbmc_large
human_pbmc_medium
human_pbmc_cpu
human_pbmc_lite
```

Potential methods:

- fewer features.
- smaller hidden layers.
- quantization where appropriate.
- distilled models if validated.

Publish resource/quality trade-offs rather than claiming one model is universally preferable.

---

# 31. Hierarchical / ensemble fallback architecture

For taxonomies with many classes, support coarse-to-fine models:

```text
Broad classifier
       |
  Immune / Stromal / Epithelial
       |
       v
Lineage classifier
       |
       v
Subtype classifier
```

This can make model size, explainability, and failure analysis more manageable.

---

# 32. Visualization

Provide optional plotting helpers:

```python
scalr.pl.prediction_umap(adata)
scalr.pl.confidence_umap(adata)
scalr.pl.uncertainty_umap(adata)
scalr.pl.marker_importance(result)
scalr.pl.confusion_matrix(metrics)
scalr.pl.calibration_curve(metrics)
```

Keep plotting dependencies optional so the core package remains lightweight.

---

# 33. CLI redesign

Add a real console entry point:

```bash
scalr validate --input data.h5ad
scalr annotate --input data.h5ad --model human_pbmc --output annotated.h5ad
scalr train --input train.h5ad --labels cell_type --group donor_id --output models/pbmc_v1
scalr explain --input annotated.h5ad --model models/pbmc_v1 --output explanations/
scalr evaluate --input test.h5ad --model models/pbmc_v1 --labels cell_type
scalr benchmark --config benchmark.yaml
scalr models list
scalr models info human_pbmc
```

Keep `python pipeline.py --config ...` working for backward compatibility during the v2 transition.

CLI requirements:

- helpful `--help`.
- clear error messages.
- non-zero exit codes on validation failure.
- machine-readable JSON output where useful.
- quiet/verbose modes.
- deterministic logging.

---

# 34. Installation and packaging redesign

The current repository recommends a specific Python environment and the installation documentation says users who install via pip may still need to obtain `pipeline.py` and config files for the complete pipeline. citeturn799194search2

The v2 package should make the common library path self-contained.

### Target installation

```bash
pip install pyscalr
```

Optional dependencies:

```bash
pip install pyscalr[analysis]
pip install pyscalr[explain]
pip install pyscalr[dev]
```

GPU installation should be documented separately instead of forcing one CUDA-specific wheel index into the base requirements.

### Packaging goals

- clear Python compatibility range.
- CPU path works from a clean environment.
- GPU instructions are explicit and versioned.
- no need to manually copy `pipeline.py`/YAML just to use the packaged API.
- support Conda installation.
- packaging smoke test in CI.

---

# 35. R / Seurat ecosystem bridge

Do not rewrite the engine in R.

Provide an interoperability layer:

```text
Seurat / SingleCellExperiment
            |
            v
     interoperable format
            |
            v
          scaLR
            |
            v
 predictions / metadata
```

This can start as documented AnnData conversion and later become a lightweight R wrapper.

---

# 36. Experiment tracking and reproducibility

Every run should create a structured artifact directory:

```text
experiment/
├── manifest.json
├── config.yaml
├── metrics.json
├── model/
├── feature_selection/
├── predictions/
├── plots/
└── logs/
```

Record:

- scaLR version.
- model version.
- Python version.
- OS.
- CPU.
- GPU.
- CUDA.
- dependency versions.
- random seeds.
- input hash.
- model hash.
- config hash.
- preprocessing hash.
- runtime.
- peak RAM/VRAM.

---

# 37. Benchmark suite

The paper establishes low-resource/scaling as a central contribution; v2 should make this a continuously reproducible benchmark rather than a one-time paper experiment. citeturn794974search2

Create:

```text
benchmarks/
├── small/
├── medium/
├── million_cells/
├── multi_million_cells/
├── cross_donor/
├── cross_platform/
├── cross_tissue/
├── rare_cell/
├── unknown_cell/
└── reproducibility/
```

Report:

### Quality

- accuracy.
- macro-F1.
- weighted-F1.
- balanced accuracy.
- per-class recall.
- AUROC where appropriate.
- confusion matrix.
- rare-cell recall.

### Reliability

- Expected Calibration Error.
- Brier score.
- unknown-cell AUROC.
- FPR95/related OOD metrics where appropriate.

### Resource usage

- wall time.
- cells/sec.
- peak RAM.
- peak VRAM.
- CPU utilization.
- GPU utilization.
- model size.

### Scaling dimensions

Vary:

- number of cells.
- number of genes.
- feature count.
- number of classes.

Publish scaling curves, not only one aggregate table.

---

# 38. Testing strategy

## Unit tests

- configuration schema.
- cross-field validation.
- gene normalization.
- gene alignment.
- preprocessing detection.
- preprocessing fit/apply separation.
- label mapping.
- calibration.
- unknown detection.
- model save/load.
- sparse matrix behavior.
- AnnData round-trip.
- resource planner.

## Integration tests

- toy end-to-end training.
- toy feature selection.
- prediction.
- explanation.
- calibration.
- artifact loading.
- CLI commands.
- checkpoint resume.

## Regression tests

Use one small frozen dataset and validate:

- prediction outputs.
- selected features.
- artifact metadata.
- basic memory budget.

## CI

GitHub Actions should run:

- lint/format.
- import smoke test.
- unit tests.
- CPU integration test.
- packaging/install test.
- documentation build where applicable.

GPU tests may be a separate workflow.

---

# 39. Logging redesign

Replace fragmented logging with one structured logger.

Every message should have:

```text
run_id
stage
severity
elapsed_time
resource_snapshot (optional)
```

Example:

```text
2026-09-21 14:10:32 | INFO | run=abc123 | stage=inference
cells=1,200,000 genes=5,000 batch=20,000 device=cpu
```

Provide console and file handlers.

---

# 40. Abstract interfaces

Introduce explicit abstract interfaces for components likely to be swapped:

```python
class BaseModel(ABC):
    ...

class BaseFeatureSelector(ABC):
    ...

class BaseTrainer(ABC):
    ...

class BasePreprocessor(ABC):
    ...

class BaseCalibrator(ABC):
    ...
```

Each should define minimal stable methods and avoid over-abstracting code that has only one implementation.

Keep backwards compatibility while refactoring.

---

# 41. Data quality and QC recommendations

Provide:

```python
scalr.qc(adata)
```

that reports, without silently modifying the data:

- low-count cells.
- high mitochondrial percentage where metadata permits.
- extreme library sizes.
- high/low feature counts.
- potential doublet signals if enabled.

Example:

```text
12,452 cells detected
3.2% low-count
1.1% high mitochondrial
0.4% unusual transcript counts

No cells were removed automatically.
```

The user decides on filtering thresholds.

---

# 42. Documentation and onboarding

Create three paths.

## Beginner

```text
10-minute quickstart
```

Only:

```python
import scalr
result = scalr.annotate(adata, model="human_pbmc")
```

## Research user

- custom training.
- feature selection.
- model explanation.
- benchmarking.
- grouped splits.

## Infrastructure/HPC user

- streaming.
- resource control.
- CLI.
- reproducible runs.
- profiling.

Add a "When should I use scaLR?" page that explains strengths and limitations honestly.

---

# 43. Clear scope: what scaLR should not try to become immediately

Do not make these the first v2 milestone:

- a universal foundation model.
- an all-purpose batch-integration package.
- a replacement for Scanpy/Seurat.
- a general-purpose probabilistic single-cell framework.

Foundation-model or generative-model integrations can be future adapters after the public API, artifact format, uncertainty layer, and benchmark framework are stable.

This keeps the project aligned with its resource-efficiency and interpretable classification identity.

---

# 44. Competitive positioning

The aim is not to claim that scaLR is universally better than every other single-cell tool.

Instead define its niche explicitly:

| Capability | scaLR 2.0 target |
|---|---|
| Low-resource inference | Core strength |
| Very large datasets | Core strength |
| Supervised annotation | Core workflow |
| Feature selection | Core workflow |
| Biomarker-oriented explanation | Core workflow |
| Uncertainty/abstention | First-class |
| Open-set detection | First-class |
| AnnData integration | First-class |
| Pretrained models | Planned/first-class |
| Generative representation learning | Not core |
| Full ecosystem replacement | Not goal |

The benchmark suite should compare documented capabilities and measured metrics, not produce a universal winner/ranking.

---

# 45. Proposed package structure

```text
scalr/
├── __init__.py
├── api.py
├── cli.py
├── result.py
│
├── config/
│   ├── schema.py
│   └── validation.py
│
├── io/
│   ├── anndata.py
│   ├── h5ad.py
│   └── streaming.py
│
├── preprocessing/
│   ├── detect.py
│   ├── normalize.py
│   ├── qc.py
│   ├── align.py
│   └── contract.py
│
├── models/
│   ├── base.py
│   ├── registry.py
│   ├── artifact.py
│   ├── inference.py
│   └── hierarchy.py
│
├── training/
│   ├── trainer.py
│   ├── sampling.py
│   ├── losses.py
│   └── splitting.py
│
├── uncertainty/
│   ├── calibration.py
│   ├── open_set.py
│   └── metrics.py
│
├── explain/
│   ├── importance.py
│   ├── markers.py
│   └── stability.py
│
├── analysis/
├── data/
├── feature/
├── nn/
│
├── visualization/
│   ├── prediction.py
│   ├── calibration.py
│   └── explanation.py
│
├── benchmark/
│   ├── runner.py
│   ├── metrics.py
│   └── resources.py
│
└── utils/
    ├── logging.py
    ├── reproducibility.py
    └── device.py
```

Do not force all existing code into this structure immediately. Migrate modules behind stable interfaces over time.

---

# 46. GitHub issues / PR roadmap

## Existing issues to retain and expand

### #2 — CI and end-to-end tests

**Goal:** establish a reliable test foundation.

Acceptance criteria:

- GitHub Actions workflow.
- unit test suite.
- toy end-to-end dataset.
- packaging/import test.
- CPU inference regression test.
- lint/format checks.

### #3 — Conda installation

Acceptance criteria:

- clean `environment.yml` or documented Conda install.
- CPU instructions.
- GPU instructions.
- clean-environment CI test.

### #4 — Config validation

Acceptance criteria:

- typed schema.
- cross-field validation.
- AnnData metadata validation.
- actionable errors.

### #5 — Unified logging

Acceptance criteria:

- one logger.
- configurable handlers.
- run/stage IDs.
- consistent severity levels.

### #6 — Abstract base classes

Acceptance criteria:

- `BaseModel`.
- `BaseTrainer`.
- `BaseFeatureSelector`.
- minimal stable interfaces.
- backwards-compatible adapters.

## New issues

### #7 — Public inference API

Implement:

- `annotate`.
- `load_model`.
- typed `PredictionResult`.

Acceptance:

- <15-line quickstart.
- AnnData support.
- CPU path.
- `device="auto"`.
- labels/confidence/probabilities.

### #8 — AnnData validation and gene alignment

Acceptance:

- duplicate/invalid genes handled.
- normalization state reported.
- feature overlap report.
- minimum-overlap threshold.

### #9 — Versioned model artifact format

Acceptance:

- manifest.
- model weights.
- preprocessing metadata.
- features.
- labels.
- calibration.
- reproducible load/save test.

### #10 — Confidence calibration and open-set detection

Acceptance:

- calibrated confidence.
- entropy.
- unknown status.
- OOD score interface.
- novel-class benchmark.

### #11 — Model registry

Acceptance:

- local list/info/download.
- metadata schema.
- compatibility validation.

### #12 — CLI

Acceptance:

- validate.
- annotate.
- train.
- evaluate.
- explain.
- benchmark.
- models.

### #13 — Sparse and bounded-memory inference

Acceptance:

- sparse test.
- chunked prediction.
- peak-memory benchmark.
- no unnecessary densification.

### #14 — Visualization and reporting

Acceptance:

- prediction UMAP.
- confidence UMAP.
- uncertainty UMAP.
- marker/explanation plot.
- calibration plot.

### #15 — Reproducible benchmark suite

Acceptance:

- frozen benchmark configs.
- quality metrics.
- resource metrics.
- scaling curves.
- environment metadata.

### #16 — Rare-class training and evaluation

Acceptance:

- class-weighted loss.
- balanced sampling.
- macro-F1/balanced accuracy.
- per-class reporting.

### #17 — Leakage-safe split and dataset checks

Acceptance:

- group-based split API.
- donor/patient/sample leakage warning.
- train/query preprocessing separation.

### #18 — Hierarchical annotation

Acceptance:

- taxonomy representation.
- coarse-to-fine prediction.
- broad/subtype levels.

### #19 — Doublet/mixed-cell screening

Acceptance:

- uncertainty/mixed-cell signal.
- explicit status.
- validation benchmark.

### #20 — Feature-selection stability

Acceptance:

- repeated feature-selection runner.
- Jaccard/rank stability.
- selection frequency report.

### #21 — Model cards and provenance

Acceptance:

- standard model-card template.
- data/model/code versioning.
- license metadata.

### #22 — Lightweight model variants

Acceptance:

- at least one lightweight model.
- measured resource/quality trade-off.

### #23 — Seurat/R interoperability

Acceptance:

- documented conversion workflow.
- round-trip labels/metadata test.

### #24 — Zarr/streaming data support

Acceptance:

- tested chunked format.
- bounded-memory benchmark.

---

# 47. Phased implementation plan

## Phase 0 — Foundation / correctness

1. #2 CI/tests
2. #3 Conda/install
3. #4 config validation
4. #5 logging
5. #6 abstract interfaces
6. #17 leakage-safe split and preprocessing separation

### Exit criteria

- clean install from scratch.
- deterministic toy training/inference.
- clear validation errors.
- CI green.

---

## Phase 1 — Public usability

7. #7 public API
8. #8 AnnData validation + gene alignment
9. #9 model artifact format
10. #12 CLI
11. AnnData output integration

### Exit criteria

```python
result = scalr.annotate(
    adata,
    model="human_pbmc",
)
```

works without YAML.

---

## Phase 2 — Scientific reliability

12. #10 confidence/calibration/open-set
13. #16 rare-class handling
14. #19 mixed/doublet screening
15. #18 hierarchical annotation
16. marker sanity checks
17. feature-selection stability #20

### Exit criteria

Results include:

- confidence.
- uncertainty.
- unknown status.
- per-class metrics.
- leakage-safe evaluation.

---

## Phase 3 — Models and distribution

18. #11 model registry
19. #21 model cards
20. automatic model selection
21. self-contained model packages
22. lightweight model variants #22

### Exit criteria

A user can download a compatible model, validate it against AnnData, run inference, inspect provenance, and reproduce the result.

---

## Phase 4 — Scale and performance

23. #13 sparse/bounded-memory inference
24. #24 Zarr/streaming
25. device/batch auto-selection
26. benchmark suite #15
27. resource profiling

### Exit criteria

The project can demonstrate scaling behavior with increasing cell counts and clear peak-RAM/VRAM measurements.

---

## Phase 5 — Ecosystem and adoption

28. #14 visualization/reporting
29. #23 R/Seurat bridge
30. optional notebook dashboard / lightweight web UI
31. tutorials and model cookbook

### Exit criteria

A Scanpy user, a CLI/HPC user, and an R/Seurat user each have a documented entry path.

---

# 48. Definition of done for scaLR 2.0

A new user should be able to:

```python
import scalr
result = scalr.annotate(adata, model="human_pbmc")
```

without writing YAML.

An advanced user should still be able to use the configuration-driven pipeline.

Every result should include:

- prediction.
- confidence.
- uncertainty.
- model/version metadata.
- preprocessing metadata.

Every model should be:

- portable.
- self-describing.
- versioned.
- provenance-aware.

Every benchmark should report both:

- scientific quality.
- computational cost.

Every installation path should work from a clean environment.

The CPU workflow must work without a CUDA-specific installation requirement.

---

# 49. Minimum viable v2 vs full v2

## MVP v2

Must-have:

- public API.
- AnnData validation.
- gene alignment.
- model artifact format.
- CPU/GPU auto-device.
- CLI.
- confidence + abstention.
- CI.
- reproducibility metadata.
- basic benchmark suite.

## Full v2

Adds:

- model hub.
- hierarchical annotation.
- cluster refinement.
- doublet screening.
- feature stability.
- lightweight models.
- Zarr/cloud.
- R bridge.
- richer dashboards.

This separation prevents the first v2 release from becoming too large.

---

# 50. Recommended first 10 PRs

### PR 1 — Packaging + CI baseline

Clean installation, test matrix, linting, CPU smoke test.

### PR 2 — Validation subsystem

`ValidationReport`, config schema, AnnData checks.

### PR 3 — Public API

`annotate`, `train`, `load_model`, `PredictionResult`.

### PR 4 — Model artifact

Self-contained manifest/weights/preprocessing/labels.

### PR 5 — Gene alignment + preprocessing contract

Reference/query compatibility.

### PR 6 — CLI

`validate`, `annotate`, `train`, `evaluate`, `explain`.

### PR 7 — Uncertainty

Calibration, entropy, margin, unknown status.

### PR 8 — Scientific evaluation

Leakage-safe splitting, class imbalance, per-class metrics.

### PR 9 — Explainability + feature stability

Cell/class explanations, repeated feature-selection stability.

### PR 10 — Benchmark harness

Resource + quality + scaling curves.

After these ten PRs, scaLR would have a materially stronger foundation before adding a model hub or cloud-scale data sources.

---

# 51. Suggested quickstart for the future README

```python
import scanpy as sc
import scalr

adata = sc.read_h5ad("pbmc.h5ad")

result = scalr.annotate(
    adata,
    model="human_pbmc",
    device="auto",
)

adata.obs["cell_type"] = result.labels
adata.obs["cell_type_confidence"] = result.confidence
adata.obs["cell_type_unknown"] = result.is_unknown

sc.pl.umap(adata, color="cell_type")
```

That should be the first code users see, followed by the advanced YAML pipeline.

---

# 52. Final strategic direction

The most valuable evolution is not to add more DNN architectures for the sake of novelty.

The opportunity is to make scaLR the following:

```text
                scaLR 2.0
                    |
       +------------+------------+
       |            |            |
      Easy       Reliable     Scalable
       |            |            |
    AnnData      uncertainty   millions+ cells
    Model Hub    open-set      low RAM
    CLI          calibration   streaming
       |            |            |
       +------------+------------+
                    |
             Explainable
                    |
           feature importance
           biomarker analysis
           stability reporting
                    |
             Reproducible
                    |
           model cards
           artifact metadata
           benchmarks
```

### Core proposition

> **scaLR is a resource-efficient, interpretable, uncertainty-aware single-cell annotation and biomarker-analysis platform designed to scale from ordinary datasets to atlas-scale workloads.**

That proposition preserves the technical contribution of the published work while addressing the practical barriers that currently prevent the project from being a broadly usable library. citeturn794974search2turn799194search2

---

# 53. Reference links

- scaLR repository: https://github.com/infocusp/scaLR citeturn799194search2
- scaLR paper: https://academic.oup.com/bib/article/26/3/bbaf243/8152766 citeturn794974search2
- CellTypist documentation (useful ecosystem reference for model-based annotation UX): https://www.celltypist.org/tutorials/onlineguide

---

## One-sentence roadmap

**First make scaLR easy to install and call, then make predictions scientifically safer, then package/distribute models, and only after that push deeper into atlas-scale streaming and ecosystem integrations.**