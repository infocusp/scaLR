from copy import deepcopy
import os
from os import path
from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection

from _scalr.analysis import build_analyser
from _scalr.analysis.evaluation import generate_and_save_classification_report
from _scalr.analysis.evaluation import get_accuracy
from _scalr.feature_extraction_pipeline import FeatureExtractionPipeline
from _scalr.nn.dataloader import build_dataloader
from _scalr.nn.model import build_model
from _scalr.utils import load_test_data_from_config
from _scalr.utils import load_train_val_data_from_config
from _scalr.utils import read_data


class EvalAndAnalysisPipeline:

    def __init__(self, analysis_config, dirpath, device):

        self.analysis_config = deepcopy(analysis_config)
        self.dirpath = dirpath
        self.device = device

        model_checkpoint = self.analysis_config.get('model_checkpoint')
        if model_checkpoint:
            model_config = read_data(
                path.join(model_checkpoint, 'model_config.yaml'))
            model_weights = path.join(model_checkpoint, 'model.pt')
            self.model, _ = build_model(model_config)
            self.model.to(self.device)
            self.model.load_weights(model_weights)
        else:
            raise Warning('''Model path not provided. Unable to perform 
                          model based analysis!''')

        # dict to transfer information between analyses
        self.primary_analysis = dict()

    def build_dataloaders(self):
        dataloader_config = deepcopy(self.analysis_config.get('dataloader'))

        self.train_dl, _ = build_dataloader(dataloader_config, self.train_data,
                                            self.target, self.mappings)

        self.val_dl, _ = build_dataloader(dataloader_config, self.val_data,
                                          self.target, self.mappings)

        self.test_dl, dataloader_config = build_dataloader(
            dataloader_config, self.test_data, self.target, self.mappings)

        self.analysis_config['dataloader'] = dataloader_config

    def load_data_and_targets_from_config(self, data_config: dict):
        """load data and targets from data config"""
        self.train_data, self.val_data = load_train_val_data_from_config(
            data_config)
        self.test_data = load_test_data_from_config(data_config)

        self.target = data_config.get('target')
        self.mappings = read_data(data_config['label_mappings'])
        self.build_dataloaders()

    def set_data_and_targets(self, train_data: Union[AnnData, AnnCollection],
                             val_data: Union[AnnData, AnnCollection],
                             test_data: Union[AnnData, AnnCollection],
                             target: Union[str, list[str]], mappings: dict):
        """Useful when you don't use data directly from config, but rather by other
        sources like feature chunking, etc.

        Args:
            train_data (Union[AnnData, AnnCollection]): training data
            val_data (Union[AnnData, AnnCollection]): validation data
            target (Union[str, list[str]]): target columns name(s)
            mappings (dict): mapping of column value to ids
                            eg. mappings[column_name][label2id] = {A: 1, B:2, ...}
        """
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.target = target
        self.mappings = mappings
        self.build_dataloaders()

    def evaluation_and_classification_report(self):
        """Evaluate the trained model and generate classification report
        on test data"""
        test_labels, pred_labels, pred_probabilities = self.model.get_predictions(
            self.test_dl, self.device)

        self.primary_analysis['test_labels'] = test_labels
        self.primary_analysis['pred_labels'] = pred_labels
        self.primary_analysis['pred_probabilities'] = pred_probabilities

        print('Accuracy: ', get_accuracy(test_labels, pred_labels))

        print('\nClassification Report:')
        generate_and_save_classification_report(
            test_labels,
            pred_labels,
            self.dirpath,
            mapping=self.mappings[self.target]['id2label'])

    def gene_analysis(self):
        gene_analysis_path = path.join(self.dirpath, 'gene_analysis')
        gene_analyser = FeatureExtractionPipeline(
            self.analysis_config.get('gene_analysis'), gene_analysis_path,
            self.device)
        gene_analyser.set_model([self.model])
        gene_analyser.set_data_and_targets(self.train_data, self.val_data,
                                           self.target, self.mappings)

        score_matrix = gene_analyser.feature_scoring()
        self.primary_analysis['score_matrix'] = score_matrix

        top_features = gene_analyser.top_feature_extraction()
        self.primary_analysis['top_features'] = top_features

    def perform_downstream_anlaysis(self):
        downstream_analysis = self.analysis_config.get('downstream_analysis',
                                                       list())
        for i, (analysis_config) in enumerate(downstream_analysis):
            analysis_config = deepcopy(analysis_config)
            analyser, analysis_config = build_analyser(analysis_config)
            downstream_analysis[i] = analysis_config

            analysis = analyser.generate_analysis(self.model, self.test_data,
                                                  self.test_dl,
                                                  **self.primary_analysis)

            # To be able to use any above anlyses in other downstream
            # Analysis
            if analysis:
                self.primary_analysis[analysis_config['name']] = analysis

        if downstream_analysis:
            self.analysis_config['downstream_analysis'] = downstream_analysis

    def get_updated_config(self):
        return self.analysis_config
