"""This is a test file for dge_lmem.py"""

import os
from os import path
import shutil

import numpy as np

from scalr.analysis import dge_lmem
from scalr.analysis.test_dge_pseudobulk import check_dge_result
from scalr.utils import generate_dummy_dge_anndata
from scalr.utils import read_data

# DgeLMEM parameters
lmem_parms_dict = {
    'fixed_effect_column': 'disease',
    'fixed_effect_factors': ['disease_x', 'normal'],
    'group': 'donor_id',
    'celltype_column': 'cell_type'
}
# Dictionary with expected results from DgeLMEM
expected_lmem_dge_result_dict = {
    'B_cell': {
        'shape': (10, 6),
        'gene': ['gene_1', 'gene_10'],
        'random_col_and_val': ('coef_disease_x', 0.3, 0.02)
    },
    'T_cell': {
        'shape': (10, 6),
        'gene': ['gene_1', 'gene_10'],
        'random_col_and_val': ('coef_disease_x', -0.05, 0.37)
    }
}


def test_lmem_generate_analysis(
        dge_parms_dict: dict = lmem_parms_dict,
        expected_dge_result_dict: dict = expected_lmem_dge_result_dict) -> None:
    """This function generates DGE result using `generate_analysis` method in DgeLMEM class.
    Finally checks the generated results with the expected by calling `check_dge_result`function.
    
    Args:
        dge_parms_dict: Parameters dictionary for the DgeLMEM class.
        expected_dge_result_dict: A dictionary with expected dge results.
    """

    os.makedirs('./tmp', exist_ok=True)

    # Generating dummy anndata.
    adata = generate_dummy_dge_anndata()

    # Path to store dge result.
    dirpath = './tmp'
    cell_subsets = list(expected_dge_result_dict.keys())
    dge_lm = dge_lmem.DgeLMEM(
        fixed_effect_column=dge_parms_dict['fixed_effect_column'],
        fixed_effect_factors=dge_parms_dict['fixed_effect_factors'],
        group=dge_parms_dict['group'],
        celltype_column=dge_parms_dict['celltype_column'],
        cell_subsets=cell_subsets)

    dge_lm.generate_analysis(adata, dirpath)
    lmem_dirpath = path.join(dirpath, 'lmem_dge_result')
    #Checking DGE result files and values for each celltype
    check_dge_result(lmem_dirpath, expected_dge_result_dict)

    shutil.rmtree('./tmp', ignore_errors=True)
