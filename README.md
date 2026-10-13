<left><img src="https://raw.githubusercontent.com/infocusp/scaLR/main/img/scaLR_logo.png" width="150" height="180"></left>

# Single-cell analysis using Low Resource (scaLR) 


[![GitHub](https://img.shields.io/github/license/InFoCusp/scaLR)](https://github.com/infocusp/scaLR?tab=GPL-3.0-1-ov-file#)
[![Documentation](https://img.shields.io/badge/docs-v1.2.0-orange)](https://infocusp.github.io/scaLR/)

[![DOI](https://zenodo.org/badge/852710658.svg)](https://doi.org/10.5281/zenodo.13767942)

## 📖 Overview 

<b>scaLR</b> is a comprehensive end-to-end pipeline that is equipped with a range of advanced features to streamline and enhance the analysis of scRNA-seq data. The major steps of the platform are:

1. <b>Data Processing</b>: Large datasets undergo preprocessing and [normalization](https://colab.research.google.com/github/infocusp/scaLR/blob/main/tutorials/preprocessing/normalization.ipynb) (if the user opts to) and are segmented into training, testing, and validation sets.

2. <b>Features Extraction</b>: A model is trained on feature subsets in a batch-wise process, so all features and samples are utilized in the feature selection process. Then, the top-k features are selected to train the final model, using a feature score based on the model's coefficients/weights or [SHAP analayis](https://colab.research.google.com/github/infocusp/scaLR/blob/main/tutorials/analysis/shap_analysis/shap_heatmap.ipynb). 

3. <b>Training</b>: A Deep Neural Network (DNN) is trained on the training dataset. The validation dataset is used to validate the model at each epoch, and early stopping is performed if applicable. Also, a [batch correction](https://colab.research.google.com/github/infocusp/scaLR/blob/main/tutorials/preprocessing/batch_correction.ipynb) method is available to correct batch effects during training in the pipeline.

4. <b>Evaluation & Downstream Analysis</b>: The trained model is evaluated using the test dataset by calculating metrics such as precision, recall, f1-score, and accuracy. Various visualizations, such as ROC curve of class annotation, feature rank plots, heatmap of top genes per class, [DGE analysis](https://colab.research.google.com/github/infocusp/scaLR/blob/main/tutorials/analysis/differential_gene_expression/dge.ipynb), and [gene recall curves](https://colab.research.google.com/github/infocusp/scaLR/blob/main/tutorials/analysis/gene_recall_curve/gene_recall_curve.ipynb), are generated.

**The below flowchart also explains the major steps of the scaLR platform.**

![image.jpg](https://raw.githubusercontent.com/infocusp/scaLR/main/img/Schematic-of-scPipeline.jpg)

scaLR has two layers: a config-free **[simple API](https://github.com/infocusp/scaLR/blob/main/docs/guide_simple_api.md)** (`scalr.train`/`scalr.annotate`/`scalr.load_model`) for quick, in-memory use, and the original **[configuration-driven pipeline](https://github.com/infocusp/scaLR/blob/main/docs/guide_pipeline.md)** (`pipeline.py` + a YAML config) for large, customized research experiments. Both share the same underlying engine.

## Pre-requisites and installation

- ScaLR supports Python 3.11-3.14. Dependencies use compatible version ranges instead of requiring one exact Python environment. New Python releases still need to be added to CI before they can be considered supported.

```
conda create -n scaLR_env python=3.12
conda activate scaLR_env
```

You can use `venv` instead of Conda:

```
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Using git:

```
git clone https://github.com/infocusp/scaLR.git
cd scaLR

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Using pip:

```
python -m pip install pyscaLR
```

If pip reports that no compatible dependency version is available, check `python --version` and upgrade pip. The package supports Python 3.11-3.14, but individual dependency releases may lag behind a newly released Python version.

**Note:** If you want to run the configuration-driven pipeline via the `pyscaLR` pip package, also clone/download `pipeline.py` and `config.yaml` from the git repository.

The base install (`pip install pyscaLR` or `pip install -r requirements.txt`) pulls the CPU build of PyTorch and works out of the box on any machine. For CUDA acceleration, install a CUDA-enabled `torch` build for your CUDA version before or after installing scaLR, following the [official PyTorch instructions](https://pytorch.org/get-started/locally/) — scaLR auto-selects the device (`device="auto"`) at train/inference time.

Contributing to scaLR itself (running tests, formatting, pre-commit hooks)? See [CONTRIBUTING.md](https://github.com/infocusp/scaLR/blob/main/CONTRIBUTING.md).

## Quickstart: annotate cells in 3 lines

```python
import scanpy as sc
import scalr

adata = sc.read_h5ad("pbmc.h5ad")

result = scalr.annotate(adata, model="models/pbmc_v1", device="auto")

adata.obs["cell_type"] = result.labels
adata.obs["cell_type_confidence"] = result.confidence
adata.obs["cell_type_unknown"] = result.is_unknown
```

To train your own model:

```python
model = scalr.train(adata, labels_key="cell_type", group_key="donor_id")
model.save("models/pbmc_v1")
```

`scalr.train` performs leakage-safe (donor-grouped) splitting, class-imbalance-aware training, confidence calibration, and reports macro-F1/balanced accuracy on held-out data. `scalr.annotate`/`model.predict` validate the input, align its genes to the model's expected feature set, and return a `PredictionResult` with calibrated `confidence`, `entropy`, `margin` and an `is_unknown` abstention flag — see [scalr/api.py](https://github.com/infocusp/scaLR/blob/main/scalr/api.py) and [scalr/result.py](https://github.com/infocusp/scaLR/blob/main/scalr/result.py).

**The [Simple API guide](https://github.com/infocusp/scaLR/blob/main/docs/guide_simple_api.md) covers the rest of this layer in depth:** data validation, gene alignment, streaming/bounded-memory prediction, self-contained model artifacts, leakage-safe evaluation, hierarchical (coarse-to-fine) annotation, cluster-aware refinement & doublet screening, explainability, dataset drift detection, feature-selection stability, model cards/registry/Hugging Face Hub, and the benchmark suite.

## Command-line interface

The simple API is also available from the shell as the `scalr` command (`validate`, `train`, `annotate`, `explain`, `evaluate`, `drift`, `models`, and `analyze` for the config-driven pipeline):

```bash
pip install -e .          # registers the `scalr` command (or: pip install pyscaLR)

scalr validate --input data.h5ad --labels-key cell_type --group-key donor_id
scalr train --input train.h5ad --labels-key cell_type --output models/pbmc_v1
scalr annotate --input query.h5ad --model models/pbmc_v1 --output annotated.h5ad
```

**See the [CLI guide](https://github.com/infocusp/scaLR/blob/main/docs/guide_cli.md) for every subcommand, its flags, and what it writes.**

## Configuration-driven pipeline

For large, customized research experiments, run the full YAML-config-driven pipeline (data ingestion/splitting/preprocessing → feature extraction → model training → evaluation & downstream analysis):

```bash
python pipeline.py --config /path/to/config.yaml -l -m
```

**See the [Pipeline guide](https://github.com/infocusp/scaLR/blob/main/docs/guide_pipeline.md)** for the input data format, how to run it, annotated example configs, and the experiment output structure, and [config/README.md](https://github.com/infocusp/scaLR/blob/main/config/README.md) for the full config parameter reference.

## Interactive tutorials

Colab notebooks covering the pipeline, DGE/gene-recall/SHAP analysis, normalization, batch correction, and a local-run example — **see [Tutorials](https://github.com/infocusp/scaLR/blob/main/docs/tutorials.md)**.

## Citation

Jogani, S., Pol, A. S., Prajapati, M., Samal, A., Bhatia, K., Parmar, J., ... & Gupta, S. (2025). scaLR: a low-resource deep neural network-based platform for single cell analysis and biomarker discovery. Briefings in Bioinformatics, 26(3), bbaf243.
