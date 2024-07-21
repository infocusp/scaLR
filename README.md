# scaLR: a low-resource deep neural network pipeline for cell types annotation and biomarker discovery

Single cell analysis using Low Resource (scaLR) is a comprehensive end-to-end pipeline that is equipped with a range of advanced features to streamline and enhance the analysis of scRNA-seq data. The major steps of the pipeline are:

1. Data processing: Large datasets undergo preprocessing and normalization (if the user opts to) and are segmented into training, testing, and validation sets. 

2. Features extraction: A model is trained on feature subsets in a batch-wise process, so all features and samples are utilised in the feature selection process. Then, the top-k features are selected to train the final model, using a feature score based on the model's coefficients/weights.

3. Training: A Deep Neural Network (DNN) is trained on the training dataset. The validation dataset is used to validate the model at each epoch and early stopping is performed if applicable.

4. Evaluation: The trained model is evaluated using the test dataset through calculating metrics such as precision, recall, f1-score, and accuracy. Various visualizations such as ROC curve of class annotation, feature rank plots, heatmap of top genes per class, DGE analysis, gene recall curves are generated.

The following flowchart explains the major steps of the scaLR pipeline.

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

If `torch not found` error pops-up if running the pipeline when installed using option 1 above, consider installing it using option 2.
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


## How to run

1. It is necessary that the user must modify the configuration file and for each stage of the pipeline is the available inside the config folder [config.yml] or [full_config.yml] as per your requirements. Simply omit/comment out stages of the pipeline you do not wish to run.
2. Config folder consists of two config file i.e. [config.yml] or [full_config.yml] and its detailed README file how to use different parameters and files.
3. Then use the `pipeline.py` file to run the entire pipeline according to your configurations. This file takes as argument the path to config (`-c | --config`), and an optional flag to log all parts of the pipelines (`-l | --log`).
4. `python pipeline.py --config /path/to/config --log` to run the scaLR. 

## Library Structure
A brief overview of the library Structure and functionalities

### [scalr](./scalr/)

- **callbacks**: 
    - `CallbackExecutor`, `EarlyStopping`, `ModelCheckpoints`, `TensorbaordLogging`
- **data**:
    - `split_data`: function to obtain and store train/test/val splits
    - `preprocess`: This function is used to normalize the data.
- **dataloader**:
    - `simple_dataloader`: generator object to prepare batch-wise data to pass through model.
- **model**:
    - `linear_model`: torch deep neural network model class
    - `shap_model`: function that use trained model for shap calculation
- **utils**:
    - `file_utils`: functions to read and write - anndata, json and yaml files

- **feature selection**:
    - `feature_chunking`: feature chunking algorithm to generate top features list  
    - `extract_top_k_features`: extract top-k features from a weight matrix using some aggregation stratergy.  
- **trainer**: `Trainer` class handles training and validation of model
- **evaluation**:
    - `get_predictions`: generate predictions of trained model on data
    - `accuracy`: generate accuracy of predictions
    - `generate_and_save_classification_report`: function to generate a classwise report containing precision, recall, f1-score metrics and to store the table
    - `perform_differential_expression_analysis`: function to generate deg analysis report, and a volcano plot of pvalues vs log2_fold_change in gene expression
    - `generate_gene_recall_curve`: function to generate gene recall curves as per user defined inputs for reference and ranked genes

### [config](./config/)
  
   - **README**: explains different parameters used to run scaLR using config files with explanation.
   - **full_config_template.yml**: a config template containing all parameters used to run scaLR and other downstream analysis
   - **config_template.yml**: a config template containing only some required parameters to run experiments.


### [examples](./examples/)
  
   - **gene_recall_curve.ipynb**: an example how to generate gene recall curve

## Pipeline Scripts (Output Structure)

- **pipeline.py**:  
Main script to run the entire pipeline.
    - `exp_dir`: root experiment directory for storage of all phases of the pipeline. Specified from the config.
        - `config.yml`: copy of config file to reproduce the experiment

- **data_ingestion.py**:  
Reads the data, and splits it into Train/Validation/Test sets for the pipeline. Then performs sample-wise normalization on the data
    - `exp_dir`
        - `data`
            - `data_split.json`: contains sample indices for train/validation/test splits
            - `train`: directory containing the train samples anndata files.
            - `val`: directory containing the validation samples anndata files.
            - `test`: directory containing the test samples anndata files.

- **feature-extractions.py**:  
Performs feature selection and extraction of new datasets containing subset features
    - `exp_dir`
        - `feature_selection`
            - `model_weights`: contains weights of each individual models trained on feature chunked data (refer to feature chunking algorithm)
            - `train`: directory containing the new feature-subsetted train samples anndatas
            - `val`: directory containing the new feature-subsetted validation samples anndatas
            - `test`: directory containing the new feature-subsetted test samples anndatas
            - `feature_class_weights.csv`: combined weights of all individual models, for each feature and class. shape: n_classes X n_features
            - `top_features.txt`: file containing list of top features selected / to be subsetted from total features.
            - `biomarkers.json`: json file containing a dictionary of classwise biomarkers/top-features.

- **train.py**:
Trains a final model on the basis of `train_datapath` and `val_datapath` in config.
    - `exp_dir`
        - `logs`: directory containing Tensorboard Logs for the training of model
        - `checkpoints`: directory containing model weights checkpointed at every interval specifief in config.
        - `best_model`: The best model checkpoint contains information to use model for inference / resume training.
            - `config.yml`: config file containing model parameters
            - `label_mappings.json`: contains mapping of class_names to class_ids used by model during training
            - `model.pt`: contains model weights
            - `model.bin`: contains model

- **evaluate.py**:  
Performs evaluation of best model trained on user defined metrics on the test set. Also performs various downstream tasks
   - `exp_dir`
        - `results`
            - `classification_report.csv`: Contains classification report showing Precision, Recall, F1, and accuracy metric for each class, on the test set.
            - `gene_recall_curves_{plot_type}.png`: Gene recall curve plots for `per_category` or `aggregated_across_all_categories` plot_type - whichever applicable or opted by user
            - `DEG_plot_{fixed_condition}_{factor_1}_vs_{factor_2}.png`: Volcano plot for DEG analysis
            - `DEG_plot_{fixed_condition}_{factor_1}_vs_{factor_2}.csv`: Differential expression values for each gene.





## Citation
Will update soon.