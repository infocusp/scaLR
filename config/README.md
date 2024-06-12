# Config Parameters

## Experiment
**device** {str}: `cuda | cpu`  
default: `cuda`  
Device to run the deep learning models on

**dirpath** {str}: `/path/to/dir`  
default: `.`  
Base directory path for all experiments

**exp_name** {str}: `experiment name`  
default: `scalr_test`  
Name of the experiment, usually based upon configurations.  
eg. scalr_ct_fs_5000_6

**exp_run** {int}: `0`  
default: `0`  
The n-th run for `exp_name`

Notes:  
Final experiment directory would be: `dirpath/{exp_name}_{exp_run}/`  
This will be considered as the root directory for this experiment run, and all subsequent I/O operations like logging, storing of files, configs, results would be done here.  

## Data
**sample_chunksize** {int}: `int | null`  
default: `10000`  
Useful for low resources utilization. This will ensure all data is stored in multiple chunks of atmost `sample_chunksize` samples. This does not hamper any logic in algorithms, but simply ensures that entire dataset is never loaded all at once on the RAM.  
`null` value will disregard this optimization.

**split_data** {dict}:  
- **split_ratio** {list [int | float]}: `[x, y, z]`  
default: `[7, 1, 2]`  
Ratio to split full data => train : val : test  
eg. There is a full dataset with 130k samples, and we want the train set with 100k, validation with 10k, and test with 20k samples. We will put up `split_ratio` as `[10, 1, 2]`. This indicates the ratios of 3 splits.  
eg. In terms of precentages if we want train set with 80%, validation with 5% and test with 15%, we can use `split_ratio` as `[80, 5, 15]`.  
- **stratify** {str}: `group | null`  
default: `null`  
Column in metadata which has multiple groups, based upon which split is stratified. This ensures that one group can belong only to one split


**normalize_samples** {bool}: `True | False`  
default: `False`  
To perform sample-wise normalization of expression values

**full_datapath** {str}: `/path/to/data`  
Full data path, will be split into train, test, and val sets.  
The data will be split into Train/Validation/Test sets and used further.  
*Optional* if  `train_datapath`, `val_datapath`, `test_datapath` parameters are given.

If `full_datapath` parameter is given, the following 3 paths will be overwritten. Otherwise they are *Required*  
**train_datapath** {str}: `/path/to/train_data`  
Training data path  
**val_datapath** {str}: `/path/to/val_data`  
Validation data path  
**test_datapath** {str}: `/path/to/test_data`  
Test data path  

**target** {str}: `target`  
Target to perform classification on. Must be present as a column_name in `adata.obs`

## Feature_Selection
**method_type** {str}: `feature_chunk`  
default: `feature_chunk`  
Algorithm to select top-k features. `feature_chunk` is the only available method right now.  

**feature_chunksize** {int}: `int | null`  
default: `3000`  
Chunks of features to subset data for training the model on iteratively.

**model** {dict}:  
Model to train the feature chunks  
- **name** {str}: `nn`  
default: `nn`  
Type of model to train each feature-chunked-subset data. Only `nn` is available right now.  
- **params** {dict}:  
--- **epochs** {int}:  
default: `25`  
Max number of epochs for to train the model.  
--- **batch_size** {int}:  
default: `15000`  
Batch_size for data loading during training. The range of this depends upon how much data can fit on GPU RAM.  
--- **lr** {float}:  
default: `1e-2`  
Learning rate during training.  
--- **weight_decay** {float}:  
default: `0.1`  
L2 Penalty during training.  

**top_features_stats** {dict}:  
Configs for extracting top-k features  
- **k** {int}:  
default: `5000`  
Number of top features to select.  
- **aggregation_strategy** {str}: `mean`  
default: `mean`  
Strategy to use model weights to give a score for each feature.  
`mean`: take the mean of absolute values of across all classes for each feature.  


## Model
**type** {str}: `linear`  
default: `linear`  
Type of model to train. Currently only `linear` model is available.  

**hyperparameters** {dict}:  
- **layers** {list [int]}: `[x, y, z]`  
*Required*  
List containing dimensions of each layer in the DNN model. The first value MUST correspond to the input features dimension, and last layer to the number of output classes.  
eg.  [10000, 1000, 10] will create a model taking input of 10000-d vectors, and create a 2-layered network having output as 10-d vector.  
- **dropout** {float}:  
default: `0`  
Dropout regularization after each layer.  
- **weights_init_zero** {bool}: `True | False`  
default: `False`  
Initialize the weights of model to zero.  

**resume_from_checkpoint** {bool}: `True | False`  
default: `False`  
To resume training from a past checkpoint  

**start_checkpoint** {str}: `/path/to/model`  
default: `null`  
*Required* if `resume_from_checkpoint` is True. Specifies the model checkpoint path to initiate weights and optimizer and start training  

## Training
**opt** {str}: `adam | sgd`  
default: `adam`  
Optimizer used for training of DNN  

**loss_fn** {str}: `log | weighted_log`  
default: `log`  
Loss function used for training of DNN.  
Weighted log is used for uneven class distribution, where weights are inverse proportions of each class size.  

**batch_size** {int}:  
default: `5000`  
Batch_size for data loading during training. The range of this size depends upon how much data can fit on GPU RAM.

**lr** {float}:  
default: `1e-3`  
Learning rate during training.  

**weight_decay** {float}:  
default: `0`  
L2 Penalty during training.  

**epochs** {int}:  
default: `100`  
Maximum number of epochs to train the model for.  

**callbacks** {dict}:  
- **tensorboard_logging** {bool}: `True | False`  
default: `True`  
Flag to enable tensorboard logging to see training curves.  
- **early_stop** {dict}:  
--- **patience** {int}:  
default: `3`  
Max number of epochs for which validation loss does not improve before stopping  
--- **min_delta** {float}:  
default: `1e-4`  
Minimum increment in validation loss to count as improvement
- **model_checkpoint** {dict}:  
--- **interval** {int}:  
default: `5`  
Intervals of epoch for which model weights are stored  

## Evaluation
**model_checkpoint** {str}: `/path/to/model`  
Uses the model checkpoint to load model for evaluation.  
*Required* if running only analysis part in pipeline.  
If training is run before evaluation, the best model path overwrites `model_checkpoint` parameter.  

**batch_size** {int}:  
default: `5000`  
Batch size for data loading onto GPU during inference. Since inference does not store additional overheads for gradient, bigger numbers can be used.  

**metrics** {list[str]}: `['accuracy', 'report', 'roc_auc', 'deg']`  
default: `['accuracy', 'report']`  
A list of evaluation metrics on the trained model.  
`accuracy`: Accuracy score for predictions on test set  
`report`: Detailed classification report showing how the model performed on each class, with recall, precision, f1-score metrics.  
`roc_auc`: store ROC-AUC plot of each class  
`deg`: perform differential gene expression analysis on data.  

**deg_config** {dict}:  
*Required* only if deg specified in `metrics`  
specific configurations to perform DEG  
- **fixed_column** {str}: *Required*  
column name in `data.obs` containing a fixed condition to subset  
- **fixed_condition** {str}: *Required*  
condition to subset data on, belonging to `fixed_column`  
- **design_factor** {str}: *Required*  
column name in `adata.obs` containing the control condition  
- **factor_categories** {str}: *Required*  
list of conditions in `design_factor` to make design matrix for  
- **sum_column** {str}: *Required*  
column name to sum values across samples  
- **fold_change** {float}:  
default: `1.5`  
fold change to filter the differentially expressed genes for volcano plot  
- **p_val** {float}:  
default: `0.05`  
p_val to filter the differentially expressed genes for volcano plot  
- **y_lim_tuple** {(float, ...)}:  
default: `null`  
values to adjust the Y-axis limits of the plot  

**gene_recall** {dict}:
Required only if user wants to generate recall curve for the experiment.
- **feature_class_weights_path**  {str}: path to feature class weights matrix. Required only if user wants to run gene recall curve explicitely after the pipeline run is over.
- **ranked_genes** {dict}:
    - **per_category** {str}: Path to csv which stores ranked genes per category of the trait.
    - **aggregated** {str}: Path to csv which stores ranked genes aggregated across all categories of the trait.
- **reference_genes** {dict}:
    - **per_category** {str}: Path to csv which stores reference genes per category of the trait.
    - **aggregated** {str}: Path to csv which stores reference genes aggregated across all categories of the trait.
- **top_K**: {int}: Top k ranked genes in which reference genes recall is to be looked for.
- **plots_per_row** {int}: Number of gene recall curve plots to plot per row in subplots.


**Note** - Few points to be kept in mind for passing ranked/reference genes list.
- If user intent to run only gene recall curve, consider commenting everything else. Just provide experiment name, run & dirpath & gene recall information in the config.
- Also, stores json file for reference genes vs its rank in pipeline generated ranked gene list per category.
- **feature_class_weights_path** when passed, is used to extract ranked genes lists for `per_category` & `aggregated` across all categories internally. So `ranked_genes` section should be omitted in that case.
- **ranked_genes** & **reference_genes** should be a csv which has columns as category names of trait and row contains ranked genes list per category.
- If **feature_class_weights_path** & **ranked_genes** section is not mentioned, then pipeline will try to look for feature_class_weights matrix at `dirpath/feature_selection/feature_class_weights.csv` path and will raise an error if matrix is not found. So user need to pass dirpath in config.yml
- **reference_genes** section is a must for generating gene recall curve.
- The csv should be in below format.
    - The index should be present in csv, as while reading we use `index_col=0`. So categores should be present in columns.
    - The reference genes csv column names should match with the actual target categoris. Please check the csv files properly.