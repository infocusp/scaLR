import os
import tqdm
import joblib
import numpy as np
import pandas as pd
import shutil


def skl_feature_chunking(train_adata, target:str, model_config:dict, chunksize:int, k:int, aggregation_strategy:str, dirpath:str):
    """ Feature selection using feature chunking approach.

        #TODO: briefly explain approach
    
        Args:
            train_data: train_dataset (anndata oject)
            target: target class which is present in dataset.obs for classifcation training
            model_config: dict containing type of model, and it related config
            chunksize: number of features to take in one training instance
            k: number of features to select from all_features
            aggregation_strategy: stratergy to aggregate features from each class, default: 'mean' 
            dirpath: directory to store all model_weights and top_features  

        Return:
            list of top k features
    """

    classes = np.unique(train_adata.obs[target])

    print('\nThe unique classes are : ', classes)

    # Load features chunkwise, train the logistic classifier model, store the model per iteration.
    os.makedirs(
        f'{dirpath}/model_weights', exist_ok = True)


    if model_config['name'] == 'logistic_classifier':
        model = LogisticRegression(**(model_config['params']))
    raise ValueError(
            '`model` must be one of [`logistic_classifier`]'
    )
    
    for start in tqdm.tqdm(range(0, len(train_adata.var_names), chunksize)):
        train_features = pd.DataFrame(
            train_adata[:, start:start + chunksize].X,
            columns=train_adata[:, start:start + chunksize].var_names)

        train_targets = train_adata[:, start:start + chunksize].obs[target]
        
        model.fit(train_features, np.ravel(train_targets))

        # Saving model per iteration.
        filename = f'{dirpath}/model_weights/Feature_chunk_{target}_model_{start}.bin'
        joblib.dump(model, filename)

    # Selecting top_k features
    print(f'Analysing features and selecting {k} features...')

    feature_class_weights = pd.DataFrame()
    model_parent_path = f'{dirpath}/model_weights'

    # Loading models from each chunk and generating feature class weights matrix.
    for file in tqdm.tqdm(os.listdir(model_parent_path)):

        if '.bin' in file:
            model_path = os.path.join(model_parent_path, file)
        else:
            continue

        model = joblib.load(model_path)
        model_stats = pd.DataFrame(model.coef_,
                                   index=model.classes_,
                                   columns=model.feature_names_in_)

        if feature_class_weights.empty:
            feature_class_weights = model_stats
        else:
            feature_class_weights = pd.concat(
                [feature_class_weights, model_stats], axis=1)

    # Stooring feature class weights matrix.
    feature_class_weights.to_csv(
        f'{dirpath}/feature_class_weights.csv'
    )

    # Aggregation strategy to be used for selecting top_k features.
    if aggregation_strategy == 'mean':
        top_features_list = feature_class_weights.abs().mean().sort_values(
            ascending=False).reset_index()['index'][:k]
    else:
        raise NotImplementedError(
            'Other aggregation strategies are not implemented yet...')

    with open(f'{dirpath}/top_features.txt','w') as fh:
        fh.write('\n'.join(top_features_list) + '\n')

    return top_features_list
    
