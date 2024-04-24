import os
import tqdm
import joblib
import numpy as np
import pandas as pd
import shutil


def feature_chunking(train_adata, target, model, chunksize, k, aggregation_strategy, filepath):
    """Feature selection using feature chunking approach."""

    classes = np.unique(train_adata.obs[target])

    print('\nThe unique classes are : ', classes)

    # Load features chunkwise, train the logistic classifier model, store the model per iteration.
    os.makedirs(
        f'{filepath}/model_weights', exist_ok = True)

    for start in tqdm.tqdm(range(0, len(train_adata.var_names), chunksize)):
        train_features = pd.DataFrame(
            train_adata[:, start:start + chunksize].X,
            columns=train_adata[:, start:start + chunksize].var_names)

        train_targets = train_adata[:, start:start + chunksize].obs[target]
        
        model.fit(train_features, np.ravel(train_targets))

        # Saving model per iteration.
        filename = f'{filepath}/model_weights/Feature_chunk_{target}_model_{start}.bin'
        joblib.dump(model, filename)

    # Selecting top_k features
    print(f'Analysing features and selecting {k} features...')

    feature_class_weights = pd.DataFrame()
    model_parent_path = f'{filepath}/model_weights'

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
        f'{filepath}/feature_class_weights.csv'
    )

    # Aggregation strategy to be used for selecting top_k features.
    if aggregation_strategy == 'mean':
        top_features_list = feature_class_weights.abs().mean().sort_values(
            ascending=False).reset_index()['index'][:k]
    else:
        raise NotImplementedError(
            'Other aggregation strategies are not implemented yet...')

    fh = open(
        f'{filepath}/top_features.txt',
        'w')
    fh.write('\n'.join(top_features_list) + '\n')
    fh.close()

    return top_features_list
    
