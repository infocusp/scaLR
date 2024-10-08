"""This is a test file for dge_pseudobulk.py"""

import os
from os import path
import shutil

import numpy as np

from scalr.analysis import dge_pseudobulk
from scalr.utils import generate_dummy_dge_anndata
from scalr.utils import read_data


def check_dge_result(result_path: str, expected_dge_result_dict: dict) -> None:
    """This function checks the expected DGE results with the generated results.

    Args:
        result_path: Path to the generated DGE results.
        expected_dge_result_dict: A dictionary with expected dge results.
    """

    celltype_list = list(expected_dge_result_dict.keys())
    result_files = os.listdir(result_path)
    csv_files = []
    svg_files = []
    for file in result_files:
        if file.endswith('.csv'):
            csv_files.append(file)
        elif file.endswith('.svg'):
            svg_files.append(file)

    # Checking for right numbers of csv & svg files.
    assert len(csv_files) == len(
        celltype_list), f"Expected {len(celltype_list)} csv files"
    assert len(svg_files) == len(
        celltype_list), f"Expected {len(celltype_list)} svg files"

    for celltype in celltype_list:
        celltype_csv = [file for file in csv_files if celltype in file]
        assert celltype_csv, f"CSV file for {celltype} is not produced"
        celltype_svg = [file for file in svg_files if celltype in file]
        assert celltype_svg, f"SVG file for {celltype} is not produced"

        celltype_df = read_data(path.join(result_path, celltype_csv[0]),
                                index_col=None)
        celltype_dge_result_dict = expected_dge_result_dict[celltype]
        assert celltype_df.shape == celltype_dge_result_dict['shape'], (
            f"There is a mismatch in the shape of the dge_result CSV file for '{celltype}'."
        )
        assert (
            celltype_df.loc[celltype_df.index[0],
                            'gene'] == celltype_dge_result_dict['gene'][0]
        ) & (np.round(
            celltype_df.loc[celltype_df.index[0],
                            celltype_dge_result_dict['random_col_and_val'][0]],
            2
        ) == celltype_dge_result_dict['random_col_and_val'][1]), (
            f'There is a mismatch in the DGE results of the first row in the CSV file for {celltype}'
        )
        assert (
            celltype_df.loc[celltype_df.index[-1],
                            'gene'] == celltype_dge_result_dict['gene'][-1]
        ) & (np.round(
            celltype_df.loc[celltype_df.index[-1],
                            celltype_dge_result_dict['random_col_and_val'][0]],
            2
        ) == celltype_dge_result_dict['random_col_and_val'][-1]), (
            f'There is a mismatch in the DGE results of the last row in the CSV file for {celltype}'
        )


# DgePseudoBulk parameters
pseudobulk_parms_dict = {
    'celltype_column': 'cell_type',
    'design_factor': 'disease',
    'factor_categories': ['disease_x', 'normal'],
    'sum_column': 'donor_id',
    'cell_subsets': ['B_cell', 'T_cell']
}
# Dictionary with expected results from DgePseudoBulk
expected_pbk_dge_result_dict = {
    'B_cell': {
        'shape': (10, 7),
        'gene': ['gene_1', 'gene_10'],
        'random_col_and_val': ('log2FoldChange', 0.97, 0.38)
    },
    'T_cell': {
        'shape': (10, 7),
        'gene': ['gene_1', 'gene_10'],
        'random_col_and_val': ('log2FoldChange', -0.26, 1.28)
    }
}


def test_pseudobulk_generate_analysis(
        dge_parms_dict: dict = pseudobulk_parms_dict,
        expected_dge_result_dict: dict = expected_pbk_dge_result_dict) -> None:
    """This function generates DGE result using `generate_analysis` method in DgePseudoBulk class.
    Finally checks the generated results with the expected by calling `check_dge_result`function.
    
    Args:
        dge_parms_dict: Parameters dictionary for the DgePseudoBulk class.
        expected_dge_result_dict: A dictionary with expected dge results.
    """

    os.makedirs('./tmp', exist_ok=True)

    # Generating dummy anndata.
    adata = generate_dummy_dge_anndata()

    # Path to store dge result.
    dirpath = './tmp'
    cell_subsets = list(expected_dge_result_dict.keys())
    dge_pbk = dge_pseudobulk.DgePseudoBulk(
        celltype_column=dge_parms_dict['celltype_column'],
        design_factor=dge_parms_dict['design_factor'],
        factor_categories=dge_parms_dict['factor_categories'],
        sum_column=dge_parms_dict['sum_column'],
        cell_subsets=cell_subsets)

    dge_pbk.generate_analysis(adata, dirpath)
    pbk_dirpath = path.join(dirpath, 'pseudobulk_dge_result')
    #Checking DGE result files and values for each celltype
    check_dge_result(pbk_dirpath, expected_dge_result_dict)

    shutil.rmtree('./tmp', ignore_errors=True)
