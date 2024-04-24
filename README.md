# Single Cell Classification Tool (unnamed)

A complete end2end pipeline and tool for scRNA-seq tabular data (cell X genes) to perform Classification tasks

## Library Structure

- scp
    - callbacks: EarlyStopping, ModelCheckpoints, Logging
    - model: Linear, Transformer
    - utils: file_utils, config
    - data: preprocessing, dataloaders
    - feature selection: feature_chunking
    - tokenizer: GeneVocab, tokenizer, padding
    - Trainer.py: training
    - evaluation.py: prediction, accuracy, report

## Data
- Currently the pipeline expects all datasets in anndata formats (`.h5ad` files only)
- The anndata object should contain cell samples as `obs` and genes as `var`.
- `adata.X` contains all gene counts/expression values.
- `adata.obs` contains any metadata regarding cells, including a column for `target` which will be used for classification. Index of `adata.obs` is cell_barcodes.
- `adata.var` contains all gene_names as Index.
- It is required to provide separate anndatas for train/validation/test sets.

## Pipeline
There are 3 independent parts in the pipeline:  
1. Top_features selection:
    - Feature Chunking: All samples are trained with a model using only a subset of features. This is iteratively done for all features. Using the coef/weights of the model to score all features available, and then selecting top_k features only to train the final model
2. Training: training a deep nueral network using train_data and validation on val_data
    - Linear Model 
3. Evaluation: Evaluating the model trained above using multiple metrics, and to perform downstream task like top-features (genes), heatmaps, etc.
    - Metrics: accuracy, classification_report
    - Downstream: top_features, heatmaps

## How to use
1. There are 3 parts in the pipeline, namely 
    - top_features selection
    - training
    - evaluation
2. All configuration customizations can be made in [`config.yml`](config.yml).
3. use the `run.sh` file to specify which parts in the pipeline to run. You can comment out whichever part of the pipeline you wish to not run. Remember to change the paths.
4. `nohup bash run.sh &` to run the pipeline in background