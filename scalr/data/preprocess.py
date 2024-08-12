from typing import Union
from os import path, walk, makedirs, remove

import numpy as np

from scalr.utils import read_data
from ..utils import write_data


def normalize_features_data(config):
    """This function takes input for normalization and calls sepcific function to perform the task.
    
    Args:
        config: config having data and normalization details.
    """
    data_config = config['data']
    train_data = read_data(data_config['train_datapath'])

    fn_name = data_config['normalize_fn']['name']
    params = data_config['normalize_fn']['params']

    if fn_name == 'standard_scale':
        normalization_fn = standard_scale(train_data, data_config, params)
    elif fn_name == 'normalize_samples':
        normalization_fn = normalize_samples(data_config, params)
    else:
        raise NotImplementedError(
            f'Specified normalization technique - {fn_name} is not implemented in pipeline!'
        )

    # Normalize the data & store it back.
    _scale_and_store_data(config, data_config['train_datapath'],
                          normalization_fn)
    _scale_and_store_data(config, data_config['val_datapath'],
                          normalization_fn)
    _scale_and_store_data(config, data_config['test_datapath'],
                          normalization_fn)

    return config


def _scale_and_store_data(config, datapath: str, normalization_fn: dict):
    """This function transforms the data using said normalization & stores back the scaled data.
    
    Args:
        config: config for extracting datapath params.
        datapath: Path to store scaled data
        normalization_fn: Data dict that stores normalization and it's parameters.
    """

    dirpath = config['dirpath']
    exp_name = config['exp_name']
    exp_run = config['exp_run']
    remove_existing_data = False
    data_type = datapath.split('/')[-1]
    expected_datapath = path.join(dirpath, f'{exp_name}_{exp_run}', 'data')

    # Making sure the scaled data is written in the expreiment directory.
    if path.join(expected_datapath, data_type) != datapath:
        store_path = path.join(expected_datapath, data_type)
        print(
            f'\nThe normalized {data_type} data will be written to path: `{store_path}`...'
        )

        makedirs(store_path, exist_ok=True)
        # Update split's data path in config.
        config['data'][f'{data_type}_datapath'] = store_path
    else:
        remove_existing_data = True
        store_path = datapath

    # Walk through each anndata file and transform as per parameters.
    for root, _, files in walk(datapath):
        for file in files:
            # Reading the chunk of anndata.
            data = read_data(path.join(root, file))
            data = data.to_memory()
            if not isinstance(data.X, np.ndarray):
                data.X = data.X.A

            # Tranforming the data.
            if normalization_fn['name'] == 'standard_scale':
                data.X = (data.X - normalization_fn['params']['mean']
                          ) / normalization_fn['params']['std']
            elif normalization_fn['name'] == 'normalize_samples':
                data.X *= (normalization_fn['params']['scaling_factor'] /
                           (data.X.sum(axis=1).reshape(len(data), 1)))

            # Removing the existing raw anndata file.
            if remove_existing_data:
                assert store_path == root, 'tvt_datapath doesn"t match with expected tvt_datapath'
                remove(path.join(root, file))

            # Writing the new scaled data file.
            write_data(data, path.join(store_path, file))


def normalize_samples(data_config, params: dict):
    """Normalize sample-wise in data

    Args:
        data: numpy array object to normalize
        scaling_factor: factor by which to scale normalized data

    Returns:
        Normalization function dict.
    """

    print('Performing cell-wise normnalisation of data...')
    normalization_fn = {
        'name': 'normalize_samples',
        'params': {
            'scaling_factor': params['scaling_factor']
        }
    }

    return normalization_fn


def standard_scale(train_data, data_config, params):
    """This function performs standard scaler on the data.
        Scaler object is fit on train data and then transformed on train, val & test data.

    Args:
        train_data: Train data AnnCollection
        data_config: config having data details
        params: normalization function parameters

    Returns:
        Normalization function dict.
    """

    print('Performing standard scaler on data...')

    batch_size = data_config['sample_chunksize']

    # Getting mean of entire train data per feature.
    if params.get('with_mean', True):
        print('--Calculating mean of data...')
        train_sum = np.zeros(train_data.shape[1]).reshape(1, -1)

        # Iterate through batches of data to get mean statistics
        for i in range(int(np.ceil(train_data.shape[0] / batch_size))):
            train_sum += train_data[i * batch_size:i * batch_size +
                                    batch_size].X.sum(axis=0)
        train_mean = train_sum / train_data.shape[0]
    else:
        # If `with_mean` is False, set train_mean to 0.
        train_mean = np.zeros(1, train_data.shape[1])

    # Getting standard deviation of entire train data per feature.
    if params.get('with_std', True):
        print('--Calculating standard deviation of data...')
        train_std = np.zeros(train_data.shape[1]).reshape(1, -1)
        # Iterate through batches of data to get std statistics
        for i in range(int(np.ceil(train_data.shape[0] / batch_size))):
            train_std += np.sum(np.power(
                train_data[i * batch_size:i * batch_size + batch_size].X -
                train_mean, 2),
                                axis=0)
        train_std /= train_data.shape[0]
        train_std = np.sqrt(train_std)

        # Handling cases where standard deviation of feature is 0, replace it with 1.
        train_std[train_std == 0] = 1
    else:
        # If `with_std` is False, set train_std to 1.
        train_std = np.ones(1, train_data.shape[1])

    # Applying standard_scaler to train, val & test data and store.
    print('--Transforming train, val & test data...')
    normalization_fn = {
        'name': 'standard_scale',
        'params': {
            'mean': train_mean,
            'std': train_std
        }
    }

    return normalization_fn
