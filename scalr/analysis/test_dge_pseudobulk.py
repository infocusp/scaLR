import os
from os import path
import shutil

import numpy as np

from scalr.analysis import dge_pseudobulk
from scalr.utils import generate_dummy_dge_anndata
from scalr.utils import read_data


def test_pseudobulk_generate_analysis():

    os.makedirs('./tmp', exist_ok=True)

    # Generating dummy anndata.
    adata = generate_dummy_dge_anndata()

    # Path to store dge result.
    dirpath = './tmp'
    dge_pbk = dge_pseudobulk.DgePseudoBulk(
        celltype_column='cell_type',
        design_factor='disease',
        factor_categories=['disease_x', 'normal'],
        sum_column='donor_id',
        cell_subsets=['B_cell', 'T_cell'])

    dge_pbk.generate_analysis(adata, dirpath)

    pbk_dirpath = dirpath + '/pseudobulk_dge_result'
    result_files = os.listdir(pbk_dirpath)
    csv_files = [file for file in result_files if file.endswith('.csv')]
    svg_files = [file for file in result_files if file.endswith('.svg')]

    # Checking for right numbers of csv & svg files.
    assert len(csv_files) == 2, "Expected 2 csv files inside 'lmem_dge_result'"
    assert len(svg_files) == 2, "Expected 2 SVG files inside 'lmem_dge_result'"
    assert any('B_cell' in file
               for file in csv_files), "csv file for B_cell not produced"
    b_cell_dge_csv = [file for file in csv_files if 'B_cell' in file][0]
    b_cell_dge_df = read_data(path.join(pbk_dirpath, b_cell_dge_csv),
                              index_col=None)
    assert b_cell_dge_df.shape == (
        10, 7
    ), "Mismatch in shape of dge_pseudobulk_result csv file, correct shape:(10,7)"
    assert (b_cell_dge_df.loc[0, 'gene'] == 'G1') & (np.round(
        b_cell_dge_df.loc[0, 'log2FoldChange'],
        2) == 0.97), 'Mismatch in dge_pseudobulk result value'
    assert (b_cell_dge_df.loc[9, 'gene'] == 'G10') & (np.round(
        b_cell_dge_df.loc[9, 'log2FoldChange'],
        2) == 0.38), 'Mismatch in dge_pseudobulk result value'
    assert any('B_cell' in file
               for file in svg_files), "svg file for B_cell not produced"
    assert any('T_cell' in file
               for file in csv_files), "csv file for T_cell not produced"
    assert any('T_cell' in file
               for file in svg_files), "svg file for T_cell not produced"

    shutil.rmtree('./tmp', ignore_errors=True)
