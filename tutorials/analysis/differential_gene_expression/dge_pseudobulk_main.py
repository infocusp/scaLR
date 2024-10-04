import argparse
import os
from os import path
import sys
from typing import Optional, Union, Tuple
import yaml

import anndata as ad
from anndata import AnnData
from anndata.experimental import AnnCollection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import scanpy as sc

from scalr.analysis import DgePseudoBulk


def main(config):
    test_data = sc.read_h5ad(config['full_datapath'], backed='r')
    dirpath = config['dirpath']
    dge_type = config['dge_type']
    assert (dge_type == 'DgePseudoBulk') and ('psedobulk_params' in config), (
        f"Check '{dge_type}' and 'psedobulk_params' in dge_config file")

    psedobulk_params = config['psedobulk_params']
    dge = DgePseudoBulk(celltype_column=psedobulk_params.get('celltype_column'),
                        design_factor=psedobulk_params['design_factor'],
                        factor_categories=psedobulk_params['factor_categories'],
                        sum_column=psedobulk_params['sum_column'],
                        cell_subsets=psedobulk_params.get('cell_subsets', None),
                        min_cell_threshold=psedobulk_params.get(
                            'min_cell_threshold', 10),
                        fold_change=psedobulk_params.get('fold_change', 1.5),
                        p_val=psedobulk_params.get('p_val', 0.05),
                        y_lim_tuple=psedobulk_params.get('y_lim_tuple', None),
                        save_plot=psedobulk_params.get('save_plot', True),
                        stdout=True)

    dge.generate_analysis(test_data, dirpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Dge analysis : psedobulk method')
    parser.add_argument('--config',
                        '-c',
                        type=str,
                        required=True,
                        help='Path to input dge_config.yaml file')
    argument = parser.parse_args()
    with open(argument.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
    main(config)
