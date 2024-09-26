import os
from os import path
import shutil

import numpy as np

from scalr.analysis import dge_lmem
from scalr.utils import generate_dummy_dge_anndata
from scalr.utils import read_data


def test_lmem_generate_analysis():

    os.makedirs('./tmp', exist_ok=True)

    # Generating dummy anndata.
    adata = generate_dummy_dge_anndata()

    # Path to store dge result.
    dirpath = './tmp'
    dge_lm = dge_lmem.DgeLMEM(fixed_effect_column='disease',
                              fixed_effect_factors=['disease_x', 'normal'],
                              group='donor_id',
                              celltype_column='cell_type',
                              cell_subsets=['B_cell', 'T_cell'])

    dge_lm.generate_analysis(adata, dirpath)

    lmem_dirpath = dirpath + '/lmem_dge_result'
    result_files = os.listdir(lmem_dirpath)
    csv_files = [file for file in result_files if file.endswith('.csv')]
    svg_files = [file for file in result_files if file.endswith('.svg')]

    # Checking for right numbers of csv & svg files.
    assert len(csv_files) == 2, "Expected 2 csv files inside 'lmem_dge_result'"
    assert len(svg_files) == 2, "Expected 2 SVG files inside 'lmem_dge_result'"
    assert any('B_cell' in file
               for file in csv_files), "csv file for B_cell not produced"
    b_cell_dge_csv = [file for file in csv_files if 'B_cell' in file][0]
    b_cell_dge_df = read_data(path.join(lmem_dirpath, b_cell_dge_csv),
                              index_col=None)
    assert b_cell_dge_df.shape == (
        10, 6
    ), "Mismatch in shape of dge_lmem_result csv file, correct shape:(10,6)"
    assert (b_cell_dge_df.loc[0, 'gene'] == 'G1') & (np.round(
        b_cell_dge_df.loc[0, 'coef_disease_x'],
        2) == 0.3), 'Mismatch in dge_lmem result value'
    assert (b_cell_dge_df.loc[9, 'gene'] == 'G10') & (np.round(
        b_cell_dge_df.loc[9, 'coef_disease_x'],
        2) == 0.02), 'Mismatch in dge_lmem result value'
    assert any('B_cell' in file
               for file in svg_files), "svg file for B_cell not produced"
    assert any('T_cell' in file
               for file in csv_files), "csv file for T_cell not produced"
    assert any('T_cell' in file
               for file in svg_files), "svg file for T_cell not produced"

    shutil.rmtree('./tmp', ignore_errors=True)
