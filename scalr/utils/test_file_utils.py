"""This is a test file for file_utils.py"""

import os
from os import path
import shutil

import numpy as np

from scalr.utils import generate_dummy_anndata
from scalr.utils import read_data
from scalr.utils import write_chunkwise_data
from scalr.utils import write_data


def test_write_chunkwise_data():
    """This function tests `write_chunkwise()`, `write_data()` & `read_data()` functions
    of file_utils."""
    os.makedirs('./tmp', exist_ok=True)

    # Generating dummy anndata.
    adata = generate_dummy_anndata(n_samples=25, n_features=5)

    # Path to write full data.
    fulldata_path = './tmp/fulldata.h5ad'
    write_data(adata, fulldata_path)

    # sample_chunksize to store full data in chunks.
    sample_chunksize = 5

    # Path to store chunked data.
    dirpath = './tmp/chunked_data/'

    # Writing fulldata in chunks.
    full_data = read_data(fulldata_path)
    write_chunkwise_data(full_data,
                         sample_chunksize=sample_chunksize,
                         dirpath=dirpath)

    # Iterating over stored chunked data to assert shape.
    observed_n_chunks = 0
    for i in range(len(os.listdir(dirpath))):
        if os.path.exists(path.join(dirpath, f'{i}.h5ad')):
            chunked_data = read_data(path.join(dirpath, f'{i}.h5ad'),
                                     backed='r')
            assert chunked_data.shape == (
                sample_chunksize, len(adata.var_names)
            ), f"There is some issue with chunk-{i}. Please check!"
            observed_n_chunks += 1
        else:
            break

    # Checking the number of chunks stored.
    expected_n_chunks = np.ceil(adata.shape[0] / sample_chunksize).astype(int)
    assert observed_n_chunks == expected_n_chunks, f"There is mismatch of observed_n_chunks - {observed_n_chunks} with expected_n_chunks - {expected_n_chunks}."

    shutil.rmtree('./tmp', ignore_errors=True)
