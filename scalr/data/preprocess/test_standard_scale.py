'''This is a test file for standard-scaler normalization.'''

from sklearn import preprocessing

from scalr.data.preprocess import standard_scale
from scalr.utils import generate_dummy_anndata


def test_fit():
    '''This function tests fit() function of sample-norm normalization.
    
    fit() function is enough for testing, as we can compare mean and std with
    sklean standard-scaler object params.
    '''

    # Creating an annadata object.
    adata = generate_dummy_anndata(n_samples=100, n_features=25)

    # Standard scaler required parameters.
    with_mean = False
    with_std = True

    # scalr standard-scale normalization.
    scalr_std_scaler = standard_scale.StandardScaler(with_mean=with_mean,
                                                     with_std=with_std)
    # Required parameter - sample_chunksize to process data in chunks.
    sample_chunksize = 4
    scalr_std_scaler.fit(adata, sample_chunksize=sample_chunksize)

    # sklearn normalization
    sklearn_std_scaler = preprocessing.StandardScaler(with_mean=with_mean,
                                                      with_std=with_std)
    sklearn_std_scaler.fit(adata.X)

    # asserts to check the calculated mean and standard deviation, the error should be less than 1e-15.
    assert sum(
        abs(scalr_std_scaler.train_mean -
            sklearn_std_scaler.mean_).flatten() < 1e-15
    ) == adata.shape[1], "Train data mean is not correctly calculated..."
    assert sum(
        abs(scalr_std_scaler.train_std - sklearn_std_scaler.scale_).flatten() <
        1e-15) == adata.shape[
            1], "Train data standard deviation is not correctly calculated..."
