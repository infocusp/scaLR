# scaLR: a low-resource deep neural network pipeline for cell types annotation and biomarker discovery

Single cell analysis using Low Resource (scaLR) is a comprehensive end to end pipeline which is equipped with a range of advanced features to streamline and enhance the analysis of scRNA-seq data. Major steps of the pipeline are: 

1. Data processing: Large datasets undergo preprocessing and normalization (if user selected) and are segmented into training, testing, and validation sets. 

2. Features extractuin: A model undergoes iterative training where all samples are utilized in each cycle, with a distinct subset of features employed in every iteration. Then the topk features are selected, to train the final model, using a feature score based on the model's coefficients/weights.

3. Training: A Deep Neural Network (DNN) is trained on the train and validation data being used as the validation set.

4. Evaluation: The trained model is evaluated using precision, recall, f1-score, and accuracy scores. Then various visualizations such as ROC curve of class annotaion, feature rank plots, per class associated common genes heatmap, DGE analysis, gene recall curves are gnerated.

![image.jpg](Schematic-of-scPipeline.jpg)

Flowchart explains scaLR major steps.

## Library Structure

- scaLR
    - callbacks: CallbackExecutor, EarlyStopping, ModelCheckpoints, TensorbaordLogging
    - model:
        - LinearModel: torch deep neural network model class
    - data:
        - split_data: function to obtain and store train/test/val splits
    - utils:
        - file_utils: functions to read and write - anndata, json and yaml files
        - config: function to load config
    - dataloader:
        - simple_dataloader: simple dataloader and generator to prepare batched data to pass through model
    - feature selection:
        - nn_feature_chunking: feature chunking algorithm to generate top features list
    - trainer: Trainer class handles training and validation of model
    - evaluation:
        - predictions: generate predictions of trained model on data
        - accuracy: generate accuracy of predictions
        - generate_and_save_classification_report: to generate a classwise report containing precision, recall, f1-score metrics and storing the table

## Pre-requisites and installation scaLR


- For smooth run of scaLAR user can create a conda environment for Python=3.9

```
conda create -n scaLR_env python=3.9

```

- Then install pytorch using below command

```
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia

OR

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```


- Last step clone the git repository and install required packages


```
pip install -r requirements.txt
```

## Input Data
- Currently the pipeline expects all datasets in [anndata](https://anndata.readthedocs.io/en/latest/tutorials/notebooks/getting-started.html) formats (`.h5ad` files only)
- The anndata object should contain cell samples as `obs` and genes as `var`.
- `adata.X` contains all gene counts/expression values.
- `adata.obs` contains any metadata regarding cells, including a column for `target` which will be used for classification. Index of `adata.obs` is cell_barcodes.
- `adata.var` contains all gene_names as Index.


## How to run

1. It is necessary user must modify the configuration file and for each stage of the pipeline in avaiable inside config folder [config.yml] or [full_config.yml] as per your requirements. Simply omit/comment out stages of the pipeline you do not wish to run.

2. Config folder consists of two config file i.e. [config.yml] or [full_config.yml] and its detailed README file how to use different parameters and files.   

3. Then use the `pipeline.py` file to run the entire pipeline according to your configurations. This file takes as argument the path to config (`-c | --config`), and an optional flag to log all parts of the pipelines (`-l | --log`).

4. `python pipeline.py --config /path/to/config --log` to run the scaLR. 

## Interactive tutorials

Please follow this link will update soon. 

## Citation
Will update soon.