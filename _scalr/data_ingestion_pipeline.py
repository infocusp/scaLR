from copy import deepcopy
import os
from os import path
from typing import Union

from _scalr.data.preprocess import build_preprocessor
from _scalr.data.split import build_splitter
from _scalr.utils import read_data
from _scalr.utils import write_data


class DataIngestionPipeline:
    """Class for Data Ingestion into the pipeline"""

    def __init__(self, data_config: dict, dirpath: str = '.'):
        """Load data config and create a `data` directory

        Args:
            data_config (dict): Data processing configuration and paths
            dirpath (str): Experiment root directory. Defaults to '.'.
        """

        self.data_config = data_config
        self.target = data_config.get('target')
        self.sample_chunksize = data_config.get('sample_chunksize')

        # Make some nessecary checks and logs
        if not self.target:
            raise Warning('Target not given')

        if not self.sample_chunksize:
            raise Warning(
                '''Sample chunksize not given. Will default to not using chunking.
                   Might results in excessive use of memory.''')

        self.datadir = path.join(dirpath, 'data')
        os.makedirs(self.datadir, exist_ok=True)

    def generate_train_val_test_split(self):
        """Function to split data into train, validation and test sets.
        """

        if self.data_config['train_val_test'].get('splitting'):
            if not self.target:
                raise Warning(
                    '''Target not provided. Will not be able to perform
                    checks regarding splits.
                    ''')

            full_datapath = self.data_config['train_val_test']['splitting'][
                'full_datapath']
            splitter_config = self.data_config['train_val_test']['splitting'][
                'splitter_config']
            splitter, splitter_config = build_splitter(splitter_config)
            self.data_config['train_val_test']['splitting'][
                'splitter_config'] = splitter_config

            # Make data splits
            train_val_test_split_indices = splitter.generate_train_val_test_split_indices(
                full_datapath, self.target)

            write_data(train_val_test_split_indices,
                       path.join(self.datadir, 'train_val_test_split.json'))

            # Check data splits
            if self.target:
                splitter.check_splits(full_datapath,
                                      train_val_test_split_indices,
                                      self.target)

            # Write data splits
            train_val_test_split_dirpath = path.join(self.datadir,
                                                     'train_val_test_split')
            os.makedirs(train_val_test_split_dirpath, exist_ok=True)
            filepaths = splitter.write_splits(full_datapath,
                                              train_val_test_split_indices,
                                              self.sample_chunksize,
                                              train_val_test_split_dirpath)

            self.data_config['train_val_test']['split_datapaths'] = filepaths

        elif self.data_config['train_val_test'].get('split_datapaths'):
            # LOG
            pass

        elif self.data_config['train_val_test'].get('final_datapaths'):
            raise ValueError(
                '''`final_datapaths` provided. User should not use `final_datapaths`
                as input to the pipeline. User input should be given only in
                `split_datapaths`. `final_datapaths` are to be generated by the pipeline
                only.
                ''')
        elif self.data_config['train_val_test'].get(
                'feature_subset_datapaths'):
            raise ValueError(
                '''Feature subset data provided. User should not use
                `feature_subset_datapaths` as input to the pipeline.
                User input should be given only in `split_datapaths`.
                `feature_subset_datapaths` are to be generated by the pipeline
                only.
                ''')
        else:
            raise ValueError('No Data Provided!')

    def preprocess_data(self):
        """Apply preprocesssing on data splits"""

        self.data_config['train_val_test']['final_datapaths'] = deepcopy(
            self.data_config['train_val_test']['split_datapaths'])
        all_preprocessings = self.data_config.get('preprocess', list())
        if not all_preprocessings: return

        data_dirpaths = dict([
            (split,
             self.data_config['train_val_test']['final_datapaths'][f'{split}'])
            for split in ['train', 'val', 'test']
        ])

        processed_data_dirpaths = [(split,
                                    path.join(self.datadir, 'processed_data',
                                              split))
                                   for split in ['train', 'val', 'test']]

        for i, (preprocess) in enumerate(all_preprocessings):
            preprocessor, preprocessor_config = build_preprocessor(preprocess)
            self.data_config['preprocess'][i] = preprocessor_config

            preprocessor.update_from_data(read_data(data_dirpaths['train']),
                                          self.sample_chunksize)

            preprocessor.process_data(data_dirpaths, self.sample_chunksize,
                                      processed_data_dirpaths)

            data_dirpaths = processed_data_dirpaths

        self.data_config['train_val_test'][
            'final_datapaths'] = processed_data_dirpaths

    def generate_mappings(self):
        """Generate an Integer mapping to and from target columns"""
        # print(self.data_config)
        column_names = read_data(self.data_config['train_val_test']
                                 ['final_datapaths']['val']).obs.columns

        datas = []
        for datapath in self.data_config['train_val_test'][
                'final_datapaths'].values():
            datas.append(read_data(datapath))

        label_mappings = {}
        for column_name in column_names:
            label_mappings[column_name] = {}

            id2label = []
            for data in datas:
                id2label += data.obs[column_name].astype(
                    'category').cat.categories.tolist()

            id2label = sorted(list(set(id2label)))
            label2id = {id2label[i]: i for i in range(len(id2label))}
            label_mappings[column_name]['id2label'] = id2label
            label_mappings[column_name]['label2id'] = label2id

        write_data(label_mappings,
                   path.join(self.datadir, 'label_mappings.json'))

        self.data_config['label_mappings'] = path.join(self.datadir,
                                                       'label_mappings.json')

    def get_updated_config(self):
        """Returns updated configs
        """
        return self.data_config
