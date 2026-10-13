# Command-line interface guide

The [simple API](guide_simple_api.md) is also available from the shell as the `scalr` command. This is separate from, and does not replace, `python pipeline.py --config ...` (see the [Pipeline guide](guide_pipeline.md) for the configuration-driven pipeline); `scalr` wraps the in-memory simple API — see [scalr/cli.py](https://github.com/infocusp/scaLR/blob/main/scalr/cli.py).

## 1. Install (registers the `scalr` command)

```bash
pip install -r requirements.txt
pip install -e .          # or: pip install pyscaLR
```

`pip install -e .` (or the packaged `pyscaLR`) registers the `scalr` console script via `[project.scripts]` in `pyproject.toml`. Verify it's on your `PATH`:

```bash
scalr --help
```

## 2. Validate your data first

```bash
scalr validate --input data.h5ad --labels-key cell_type --group-key donor_id
```

Prints the validation report (structure/numerical/label checks) and exits non-zero if the data is unusable — useful as a pre-flight check or CI gate.

## 3. Train a model

```bash
scalr train \
  --input train.h5ad \
  --labels-key cell_type \
  --group-key donor_id \
  --output models/pbmc_v1 \
  --epochs 15 \
  --device auto
```

Writes a self-contained model artifact directory to `models/pbmc_v1` (weights, manifest, calibration, a `README.md` model card, etc.) and prints macro-F1/balanced accuracy on the held-out split.

## 4. Annotate new data with it

```bash
scalr annotate \
  --input query.h5ad \
  --model models/pbmc_v1 \
  --output annotated.h5ad \
  --device auto \
  --flag-doublets
```

Writes predictions (`scalr_pred`, `scalr_confidence`, `scalr_unknown`, etc.) into `annotated.h5ad`'s `.obs`/`.obsm`/`.uns`. `--model` also accepts a name registered in the local model registry, not just a path. Add `--streaming --chunk-size 20000` for large inputs to gene-align and score in bounded-memory chunks instead of all at once.

## 5. Explain predictions

```bash
scalr explain --input query.h5ad --model models/pbmc_v1 --indices 10 20 30 --top-k 20
scalr explain --model models/pbmc_v1 --class-name T_cell --top-k 50
```

Prints, per requested cell, the genes whose expression most supported or contradicted the prediction; `--class-name` explains a class in general instead of specific cells.

## 6. Evaluate against labeled test data

```bash
scalr evaluate --input test.h5ad --model models/pbmc_v1 --labels-key cell_type
```

Prints macro-F1/weighted-F1/balanced-accuracy plus a per-class precision/recall/F1 table.

## 7. Check for dataset drift

```bash
scalr drift --input query.h5ad --model models/pbmc_v1 --fail-on-drift
```

Compares `query.h5ad` against the model's training-time gene-expression and library-size statistics and prints a drift report; `--fail-on-drift` exits non-zero if drift is detected, for use as a pre-flight/CI gate before trusting predictions on a new cohort.

## 8. Manage the local model registry & Hugging Face Hub

```bash
scalr models list
scalr models info human_pbmc
scalr models push models/pbmc_v1 your-org/human-pbmc   # requires: pip install "pyscaLR[hub]"
scalr models pull your-org/human-pbmc                    # downloads and registers locally
```

Registering a model into the registry (`scalr.models.register(...)`) is currently Python-API-only, not yet a CLI subcommand.

## 9. Run downstream analyses

Configure the desired analyses under `analysis` in a YAML file. Supported analyses include `Heatmap`, `RocAucCurve`, `GeneRecallCurve`, `DgePseudoBulk`, and `DgeLMEM`.

Run them through the CLI:

```bash
scalr analyze \
    --config config/config.yaml \
    --log
```

Optional flags:

```bash
--level INFO
--logpath scalr_experiments/analysis.log
--memoryprofiler
```

The command uses the existing configuration-driven pipeline and writes results under the `experiment.dirpath` configured in the YAML file.
