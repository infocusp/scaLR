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
- Currently the pipeline expects all datasets in [anndata](https://anndata.readthedocs.io/en/latest/tutorials/notebooks/getting-started.html) formats (`.h5ad` files only)
- The anndata object should contain cell samples as `obs` and genes as `var`.
- `adata.X` contains all gene counts/expression values.
- `adata.obs` contains any metadata regarding cells, including a column for `target` which will be used for classification. Index of `adata.obs` is cell_barcodes.
- `adata.var` contains all gene_names as Index.

## Pipeline
There are 3 independent parts in the pipeline:  
1. Top_features selection: A model undergoes iterative training where all samples are utilized in each cycle, with a distinct subset of features employed in every iteration. Then the top_k features are selected, to train the final model, using a feature score based on the model's coefficients/weights.
2. Training: A Deep Neural Network (DNN) is trained on the train_data with val_data being used as the validation set.
3. Evaluation: The trained model is evaluated using precision, recall, f1-score, and accuracy scores. Then various visualizations like feature rank plots (genes) and heatmaps are prepared.

## How to use
1. Modify the configuration and each stage of the pipeline in [config.yml](config.yml) as per your requirements. Simply omit / comment out stages of the pipeline you do not wish to run.
2. use the `pipeline.py` file to run the entire pipeline according to your configurations. This file takes as argument the path to config (`-c | --config`), and an optional flag to log all parts of the pipelines (`-l | --log`).
3. `python pipeline.py --config /path/to/config --log` to run the pipeline