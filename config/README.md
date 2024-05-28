
# Config Parameters

## Experiment
**device** {str}: `cuda` | `cpu`
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
Experiment runs for comparison

Notes:
Final experiment directory would be: `dirpath/{exp_name}_{exp_run}/`
This will be considered as the root directory for this run, and all subsequent file operations, logging, storing would be done here. Any subsequent occurrence of `dirpath` should be assumed as the root directory and not parameter passed.

## Data
**chunksize** {int}: `number` | `null`
default: `10000`
Useful for low resources utilization. This will ensure all data is stored in multiple chunks of atmost `chunksize` samples. This does not hamper any logic in algorithms, but simply ensures that entire dataset is never loaded all at once on the RAM.
`null` value will disregard this optimization. 

**split_data** {dict}:
- **split_ratio** {list [int | float]}: `[x, y, z]`
default: `[7, 1, 2]`
Ratio to split full data => train : val : test
- **stratify** {str}: `group | null`
default: `null`
Column in metadata which has multiple groups, based upon which split is stratified. This ensures that one group can belong only to one split


**normalize_data** {bool}: `True | False`
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

Notes:
TODO

## Feature Selection
**device** {str}: `cuda` | `cpu`
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
Experiment runs for comparison
feature_selection:

    # [Optional] path to weight matrix containing weights for all features across all classes
    # If path specified, feature importance algorithm will not work, and directly use weights to perform selection
    # weight_matrix: 'full_test_0/feature_selection/feature_class_weights.csv'

    chunksize: 5000
    # [Required] chunks of features to train the model on iteratively

    method_type: feature_chunk # Only method availble now

    # [Required] [name: logistic_classifier/nn]
    # logistic regression: keyword args of logistic regression classifier of sklearn
    #                      can be passes in params
    # nn: params are optional, can include epochs, batch_size, lr- learning rate, l2- L2 weight penalty
    model:
      name: nn
      params:
          epochs: 1

    # [Required]
    # k: number of top features to extract
    # aggregation_strategy: stratergies to obtain top features. only mean implemented now
    top_features_stats:
        k: 3000
        aggregation_strategy: mean # [Required][mean/class weighted density/top (k/n_classes) per class]

    # Will store extracted features subset on disk
    # the storage will be according to data=>chunksize parameter
    store_on_disk: True
