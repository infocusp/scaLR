# Config Parameters

This README explains different parameters used to run scaLR using config file with explanation of different subsection.

**NOTE**: Majority of parameters defined below follow name-params declaration i.e. you just need to mention class name in `name` that has to be defined in codebase & its parameters under `params` dict.


## Experiment
**device** {str}: `cuda | cpu`  
default: `cuda`  
Device to run the deep learning models on

**dirpath** {str}: `/path/to/dir`  
default: `.`  
Base directory path for all experiments

**exp_name** {str}: `experiment name`  
default: `scalr_test`  
Name of the experiment, usually based upon configurations and user choice.
eg. scalr_ct_fs_5000_6

**exp_run** {int}: `0`  
default: `0`  
The n-th run for `exp_name`

*Notes*:  
Final experiment directory would be: `dirpath/{exp_name}_{exp_run}/`  
This will be considered as the root directory for this experiment run, and all subsequent I/O operations like logging, storing of files, configs, results would be save inside this folder.


## Data
This section(`data` in config) focuses on data loading using specified datapaths, splitting into train/val/test & preprocess them using mentioned preprocessing techniques.

**sample_chunksize** {int}: `int | null`  
default: `50000`  
Useful for low resources utilization. This will ensure all data is stored in multiple chunks of atmost `sample_chunksize` samples. This does not hamper any logic in algorithms, but simply ensures that entire dataset is never loaded all at once on the RAM.  
`null` value will disregard this optimization.

**train_val_test** {dict}:  
This section splits the data using mentioned splitting technique mentioned in `splitter_config` & required params like `split_ratio` and `straitify` option. Example below.

    splitter_config:
        name: GroupSplitter
        params:
            split_ratio: [7, 1, 2.5]
            stratify: 'donor_id'

- Currently platform supports only `StratifiedSplitter` for splitting the data.


**split_datapaths** {str}:  
If you already have data splits - `train`, `val` & `test` directories then you can mention their parent directory path in `split_datapaths` and remove above `splitter_config`.


**preprocess** {dict}:  
You can mention the normalization techniques and it's required parameters available in platform or create your custom normalization in library and use it here. You can mention list of transformation as well. Example below

    preprocess:
        - name: SampleNorm
          params:
            scaling_factor: 1.0

        - name: StandardScaler
          params: 
            with_mean: True
            with_std: False

- You can opt for uni or multi-transformations. Above example will apply `SampleNorm` normalization first and then `StandardScaler` normalization. 

- Currently platform supports `SampleNorm` and `StandardScaler` for normalizations.


**target** {str}: `target`  
Target to perform classification on. Must be present as a column_name in `adata.obs`


## Feature Selection
This section(`feature_selection` in config) focuses on feature subsetting, training on each subset and feature selection at the end.


**scores**: `/path/to/score_matrix`. [Optional]  
If mentioned, the feature subsetting process will be skipped and this matrix will be used to extract top features based on `feature_selector` config.

**feature_subsetsize** {int}: `int | null`  
default: `5000`  
- Chunks of features to subset data for training the model iteratively.

**model** {dict}:  
Mention the name of model class in `name` & its required parameters under `params` dict. Example below

    model:
        name: SequentialModel
        params:
            layers: [5000, 6]
            weights_init_zero: True

- Currently platform supports `SequentialModel` for model training.

**model_train_config** {dict}:  
Mention the mode training parameters like `trainer`, `dataloader`, `optimizer`, `loss` functions, epoch etc. those are required parameters explained below:

    model_train_config:
        trainer: SimpleModelTrainer

        dataloader: 
            name: SimpleDataLoader
            params:
                batch_size: 25000
                padding: 5000
        
        optimizer:
            name: SGD
            params:
                lr: 1.0e-3
                weight_decay: 0.1

        loss:
            name: CrossEntropyLoss
        
        epochs: 1

- Current support/s in the platform for following model configurations is/are:
    - trainer: `SimpleModelTrainer`
    - dataloader: `SimpleDataLoader|SimpleMetaDataLoader`
    - loss: pytorch supported all inbuilt loss functions. The class name should match exactly with torch.nn class names.


**scoring_config** {dict}:  
Mention the scorer config example given below

    scoring_config: 
        name: LinearScorer


- Currently platform supports `LinearScorer` and `ShapScorer` for scoring the features.


**features_selector** {dict}:  
Mention the feature aggregation strategy and its required parameters explained below

    features_selector:
        name: AbsMean
        params:
            k: 5000

- Currently platform supports for `feature_selectors`:
    - `AbsMean` - abs followed by mean across all class types
    -  `ClasswisePromoters` - top K promoters per class
    -  `ClasswiseAbs` - top K from promoters and inhibitors combined


## Final Training
This section(`final_training` in config) focuses on model training on the data.

**model** {dict}:  
Required parameter similar to as mentioned in `feature_selection` section.

**model_training_config** {dict}:  
Required parameter similar to as mentioned in `feature_selection` section.

**callbacks** {list}:  
In this section, you can define callbacks as per your requirement.

    callbacks:
        - name: TensorboardLogger
        - name: EarlyStopping
            params:
            patience: 3
            min_delta: 1.0e-4
        - name: ModelCheckpoint
            params:
            interval: 5

- Currently platform supports `TensorboardLogger`, `EarlyStopping` & `ModelCheckpoint` for callbacks.

## Analysis
This section(`analysis` in config) focuses on gene analysis like biomarker identification and further downstream analysis like SHAP, gene recall, DGE etc.

**model_checkpoint** {str}:  `/path/to/best_model` [Optional]  
Path to load the model from. If your model resides under say `best_model/` directory, then you need to mention path till `best_model/`. During end-to-end platform run, it is not required. It is required when you are performing analysis independently after model training.

**dataloader** {dict}:  
The dataloader has to be exactly similar which was used during final training of the model. Example in `feature_selection` part.

**gene_analysis** {dict}:  
                name: ShapScorer
This section consists of scparametersorers and feature selectors information, as seen in `feature_selection` section as well. The `scoring_config` will score the gene for each class type and `feature_selector` will extract top biomarkers for each class type as per mentioned strategy. Explained below are the 2 options available for `scoring_config`.

1. Gene analysis using LinearScorer (Default)

        gene_analysis:
            scoring_config:
                name: LinearScorer


2. Gene analysis using ShapScorer (User specific optional)


        gene_analysis:
            scoring_config:
                name: ShapScorer
                params:
                    dataloader:
                        name: SimpleMetaDataLoader
                        params:
                            batch_size: 10
                            padding: 5000
                            metadata_col: ['Cell_Type']

                    top_n_genes: 100
                    background_tensor: 20
                    samples_abs_mean: True
                    early_stop:
                        patience: 2
                        threshold: 95


    **feature_selector** {dict}:
    Mention the feature aggregation strategy and its required parameters explained below

            features_selector:
                name: ClasswisePromoters
                params:
                    k: 100



**downstream_analysis** {list}:  
This section performs gene analysis task such as generating classification reports, extracting biomarkers for each class and downstream analysis tasks like Heatmap of k genes across different cell types, generate ROC AUC curve, Gene recall curve and Differential gene expression analysis. You can mention list of name-params for tasks you want perform. Example below
    
    downstream_analysis
        - name: Heatmap
          params:
              top_n_genes: 100

        - name: RocAucCurve

        - name: GeneRecallCurve
            params:
            reference_genes_path: '/path/to/reference_genes.csv'
            top_K: 300
            plots_per_row: 3
            features_selector:
                name: ClasswiseAbs
                params: {}

        - name: DgePseudoBulk
            params:
            celltype_column: 'cell_type'
            design_factor: 'disease'
            factor_categories: ['COVID-19', 'normal']
            sum_column: 'donor_id'
            cell_subsets: ['non-classical monocyte','natural killer cell']
            min_cell_threshold: 10
            fold_change: 1.5
            p_val: 0.05
            save_plot: True
            top_K: 300
            plots_per_row: 3

        - name: DgeLMEM
            params:
            fixed_effect_column: 'disease'
            fixed_effect_factors: ['COVID-19', 'normal']
            group: 'donor_id'
            celltype_column: 'cell_type'
            cell_subsets: ['non-classical monocyte','natural killer cell']min_cell_threshold: 10
            n_cpu: 6
            gene_batch_size: 1000
            coef_threshold: 0
            p_val: 0.05
            save_plot: True
