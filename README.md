<p align="center"><img src="scaLR_logo.png" width="125" height="70"></p>

# scaLR: a low-resource deep neural network platform for cell types analysis and biomarker discovery

Single cell analysis using Low Resource (scaLR) is a comprehensive end-to-end pipeline that is equipped with a range of advanced features to streamline and enhance the analysis of scRNA-seq data. The major steps of the platform are:

1. Data processing: Large datasets undergo preprocessing and normalization (if the user opts to) and are segmented into training, testing, and validation sets.

2. Features extraction: A model is trained on feature subsets in a batch-wise process, so all features and samples are utilised in the feature selection process. Then, the top-k features are selected to train the final model, using a feature score based on the model's coefficients/weights.

3. Training: A Deep Neural Network (DNN) is trained on the training dataset. The validation dataset is used to validate the model at each epoch and early stopping is performed if applicable. Also, batch correction method is available to correct batch effects during training in the pipeline.

4. Evaluation: The trained model is evaluated using the test dataset through calculating metrics such as precision, recall, f1-score, and accuracy. Various visualizations such as ROC curve of class annotation, feature rank plots, heatmap of top genes per class, DGE analysis, gene recall curves are generated.

The following flowchart explains the major steps of the scaLR platform.

![image.jpg](Schematic-of-scPipeline.jpg)

## Pre-requisites and installation scaLR


- ScalR can be installed using Conda or pip. It is tested for Python 3.9 at this moment.

```
conda create -n scaLR_env python=3.9

```

- install pytorch using the below command

```
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia

```

OR

```
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

If `torch not found` error pops-up if running the platform when installed using option 1 above, consider installing it using option 2.
```


- Last step is to clone the git repository and install required packages by activating the conda env


```
conda activate scaLR_env

pip install -r requirements.txt

```

## Input Data
- Currently the pipeline expects all datasets in [anndata](https://anndata.readthedocs.io/en/latest/tutorials/notebooks/getting-started.html) formats (`.h5ad` files only)
- The anndata object should contain cell samples as `obs` and genes as `var`.
- `adata.X` contains all gene counts/expression values.
- `adata.obs` contains any metadata regarding cells, including a column for `target` which will be used for classification. Index of `adata.obs` is cell_barcodes.
- `adata.var` contains all gene_names as Index.


## Platform Scripts (Output Structure)
**pipeline.py**:
Main script to run the entire pipeline.
    - `exp_dir`: root experiment directory for storage of all phases of the pipeline. Specified from the config.
    - `config.yml`: copy of config file to reproduce the experiment

- **data_ingestion**:
Reads the data, and splits it into Train/Validation/Test sets for the pipeline. Then performs sample-wise normalization on the data
    - `exp_dir`
        - `data`
            - `train_val_test_split.json`: contains sample indices for train/validation/test splits
            - `label_mappings.json`: contains mappings of all metadata columns between labels and ids
            - `train_val_test_split`: directory containing the train, validation and test samples anndata files.

- **feature_extraction**:
Performs feature selection and extraction of new datasets containing subset features
    - `exp_dir`
        - `feature_extraction`
            - `chunked_models`: contains weights of each individual models trained on feature chunked data (refer to feature chunking algorithm)
            - `feature_subset_data`: directory containing the new feature-subsetted train, val and test samples anndatas
            - `score_matrix.csv`: combined scores of all individual models, for each feature and class. shape: n_classes X n_features
            - `top_features.json`: file containing list of top features selected / to be subsetted from total features.

- **final_model_training**:
Trains a final model on the basis of `train_datapath` and `val_datapath` in config.
    - `exp_dir`
        - `model`
            - `logs`: directory containing Tensorboard Logs for the training of model
            - `checkpoints`: directory containing model weights checkpointed at every interval specifief in config.
            - `best_model`: The best model checkpoint contains information to use model for inference / resume training.
                - `config.yml`: config file containing model parameters
                - `label_mappings.json`: contains mapping of class_names to class_ids used by model during training
                - `model.pt`: contains model weights

- **eval_and_analysis**:
Performs evaluation of best model trained on user defined metrics on the test set. Also performs various downstream tasks
   - `exp_dir`
        - `analysis`
            - `classification_report.csv`: Contains classification report showing Precision, Recall, F1, and accuracy metric for each class, on the test set.
            - `gene_recall_curve.svg`: Contains gene recall curve plots.
            - `gene_recall_curve_info.json`: Contains reference genes list which are present in top_K ranked genes per class for each model.
            - `gene_analysis`
                - `score_matrix.csv`: score of final model, for each feature and class. shape: n_classes X n_features
                - `top_features.json`: file containing list of top features selected / biomarkers
            -  `heatmaps`
                - `class_name.svg`: Contains heatmaps for each class type. E.g. B.svg
            - `roc_auc.svg`: Contains ROC-AUC plot


## How to run

1. It is necessary that the user must modify the configuration file and for each stage of the pipeline is the available inside the config folder [config.yml] or [full_config.yml] as per your requirements. Simply omit/comment out stages of the pipeline you do not wish to run.
2. Refer config.yml & it's detailed config [README](config_README.md) file on how to use different parameters and files.
3. Then use the `pipeline.py` file to run the entire pipeline according to your configurations. This file takes as argument the path to config (`-c | --config`), and an optional flag to log all parts of the pipelines (`-l | --log`).
4. `python pipeline.py --config /path/to/config -c config.yaml -l` to run the scaLR.


## Interactive tutorials
Detailed tutorials have been made on how to use some functionalities as a scaLR library. Find links below.

- Normalization - `tutorials/preprocessing/normalization.ipynb`
- Batch correction - `tutorials/preprocessing/batchc_correction.ipynb`
- Gene recall curve - `tutorials/analysis/gene_recall_curve/gene_recall_curve.ipynb`
- Differential gene expression analysis - `tutorials/analysis/differential_gene_expression/dge.ipynb`
- SHAP analysis - `tutorials/analysis/shap_analysis/shap_heatmap.ipynb`


## Citation
Will update soon.