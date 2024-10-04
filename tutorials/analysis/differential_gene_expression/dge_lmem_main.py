import argparse
import multiprocessing
import os
from os import path
import pickle
import resource
import string
import sys
from typing import Optional, Union, Tuple
import traceback
import warnings
import yaml

from anndata import AnnData
from anndata import ImplicitModificationWarning
import anndata as ad
from anndata.experimental import AnnCollection
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
import scanpy as sc
from scipy.optimize import OptimizeWarning
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import HessianInversionWarning, ConvergenceWarning

from scalr.analysis import DgeLMEM


def main(config):
    test_data = sc.read_h5ad(config['full_datapath'], backed='r')
    dirpath = config['dirpath']
    dge_type = config['dge_type']
    assert (dge_type == 'DgeLMEM') and ('lmem_params' in config), (
        f"Check '{dge_type}' and 'lmem_params' in dge_config file")

    lmem_params = config['lmem_params']
    dge = DgeLMEM(fixed_effect_column=lmem_params['fixed_effect_column'],
                  fixed_effect_factors=lmem_params['fixed_effect_factors'],
                  group=lmem_params['group'],
                  celltype_column=lmem_params.get('celltype_column', None),
                  cell_subsets=lmem_params.get('cell_subsets', None),
                  min_cell_threshold=lmem_params.get('min_cell_threshold', 10),
                  n_cpu=lmem_params.get('n_cpu', 6),
                  gene_batch_size=lmem_params.get('gene_batch_size', 1000),
                  coef_threshold=lmem_params.get('coef_threshold', 0),
                  p_val=lmem_params.get('p_val', 0.05),
                  y_lim_tuple=lmem_params.get('y_lim_tuple', None),
                  save_plot=lmem_params.get('save_plot', True),
                  stdout=True)

    dge.generate_analysis(test_data, dirpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dge analysis : LMEM method')
    parser.add_argument('--config',
                        '-c',
                        type=str,
                        required=True,
                        help='Path to input dge_config.yaml file')
    argument = parser.parse_args()
    with open(argument.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
    main(config)
