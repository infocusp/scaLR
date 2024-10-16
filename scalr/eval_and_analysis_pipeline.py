"""This file contains an implementation of model evaluation and performs downstream analysis tasks."""

from copy import deepcopy
import os
from os import path
from typing import Union

from anndata import AnnData
from anndata.experimental import AnnCollection

from scalr.analysis import build_analyser
from scalr.analysis.evaluation import generate_and_save_classification_report
from scalr.analysis.evaluation import get_accuracy
from scalr.feature_extraction_pipeline import FeatureExtractionPipeline
from scalr.nn.dataloader import build_dataloader
from scalr.nn.model import build_model
from scalr.utils import FlowLogger
from scalr.utils import load_full_data_from_config
from scalr.utils import load_test_data_from_config
from scalr.utils import load_train_val_data_from_config
from scalr.utils import read_data


class EvalAndAnalysisPipeline:
    """Class for evaluation and analysis of the trained model."""

    def __init__(self, analysis_config, dirpath, device):
        """Initialize required parameters for analysis.
        
        Args:
            analysis_config: Analysis config.
            dirpath: Path to store analysis outputs.
        """
        self.flow_logger = FlowLogger('Eval&Analysis')

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
            self.flow_logger.warning(
                'Model path not provided. Unable to perform model-based analysis!'
            )
            self.model = None

        # Dict to transfer information between analyses.
        self.primary_analysis = dict()

    def build_dataloaders(self):
        """A function to build dataloader for train, validation, and test data."""
        dataloader_config = deepcopy(self.analysis_config.get('dataloader'))

        if not dataloader_config:
            self.flow_logger.warning('DataLoader configs not provided!')
            self.train_dl = None
            self.val_dl = None
            self.test_dl = None
            return

        self.train_dl, _ = build_dataloader(dataloader_config, self.train_data,
                                            self.target, self.mappings)

        self.val_dl, _ = build_dataloader(dataloader_config, self.val_data,
                                          self.target, self.mappings)

        self.test_dl, dataloader_config = build_dataloader(
            dataloader_config, self.test_data, self.target, self.mappings)

        self.analysis_config['dataloader'] = dataloader_config

    def load_data_and_targets_from_config(self, data_config: dict):
        """A function to load data and targets from data config.

        Args:
            data_config: Data config.
        """
        self.train_data, self.val_data = load_train_val_data_from_config(
            data_config)
        self.test_data = load_test_data_from_config(data_config)
        self.full_data = load_full_data_from_config(data_config)

        self.target = data_config.get('target')
        self.mappings = read_data(data_config['label_mappings'])
        self.build_dataloaders()

    def set_data_and_targets(self, train_data: Union[AnnData, AnnCollection],
                             val_data: Union[AnnData, AnnCollection],
                             test_data: Union[AnnData, AnnCollection],
                             target: Union[str, list[str]], mappings: dict):
        """A function to set data when you don't use data directly from config,
        but rather by other sources like feature subsetting, etc.

        Args:
            train_data (Union[AnnData, AnnCollection]): Training data.
            val_data (Union[AnnData, AnnCollection]): Validation data.
            target (Union[str, list[str]]): Target columns name(s).
            mappings (dict): Mapping of a column value to ids
                            eg. mappings[column_name][label2id] = {A: 1, B:2, ...}.
        """
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.full_data = None
        self.target = target
        self.mappings = mappings
        self.build_dataloaders()

    def evaluation_and_classification_report(self):
        """A function to evaluate the trained model and generate classification report on test data."""
        self.flow_logger.info(
            'Calculating accuracy and generating classification report on test set'
        )

        test_labels, pred_labels, pred_probabilities = self.model.get_predictions(
            self.test_dl, self.device)

        self.primary_analysis['test_labels'] = test_labels
        self.primary_analysis['pred_labels'] = pred_labels
        self.primary_analysis['pred_probabilities'] = pred_probabilities
        self.primary_analysis['mapping'] = self.mappings[
            self.target]['id2label']

        accuracy = get_accuracy(test_labels, pred_labels)

        # Generate classification report.
        generate_and_save_classification_report(
            test_labels,
            pred_labels,
            self.dirpath,
            mapping=self.mappings[self.target]['id2label'])

    def gene_analysis(self):
        """A function to perform analysis on trained model to get top genes and biomarkers."""

        self.flow_logger.info('Performing gene analysis')
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

    def _perform_downstream_analysis(self, samples: str):
        """A function to perform all downstream analysis tasks on model and data.
        
        Args:
            samples ['test' | 'full']: indicates the samples to perform downstream 
                                       analysis.
        """
        downstream_analysis = self.analysis_config.get(
            f'{samples}_samples_downstream_analysis', list())
        if downstream_analysis:
            self.flow_logger.info(
                f'Performing Downstream Analysis on {samples} samples')
            samples_analysis_path = path.join(self.dirpath,
                                              f'{samples}_samples')
            os.makedirs(samples_analysis_path, exist_ok=True)

        for i, (analysis_config) in enumerate(downstream_analysis):
            try:
                self.flow_logger.info(f'Performing {analysis_config["name"]}')

                analysis_config = deepcopy(analysis_config)
                analyser, analysis_config = build_analyser(analysis_config)
                downstream_analysis[i] = analysis_config

                model = self.model if samples == 'test' else None
                dl = self.test_dl if samples == 'test' else None
                data = self.test_data if samples == 'test' else self.full_data

                analysis = analyser.generate_analysis(
                    model=model,
                    test_data=data,
                    test_dl=dl,
                    dirpath=samples_analysis_path,
                    **self.primary_analysis)

                # To be able to use any above analyses in other downstream analysis.
                if analysis:
                    self.primary_analysis[analysis_config['name']] = analysis
            except Exception as e:
                self.flow_logger.exception(e)

        if downstream_analysis:
            self.analysis_config[
                f'{samples}_samples_downstream_analysis'] = downstream_analysis

    def full_samples_downstream_anlaysis(self):
        """A function to perform downstream analysis tasks on all samples data.
        
        Note: The Model & DataLoader will not be passsed since it is assumed that
        a model is trained on the train data, so analysis by model should not
        be on full samples data.
        """
        self._perform_downstream_analysis('full')

    def test_samples_downstream_anlaysis(self):
        """A function to perform downstream analysis tasks on model and test 
        samples data.
        """
        self._perform_downstream_analysis('test')

    def get_updated_config(self) -> dict:
        """A function to return updated configs."""
        return self.analysis_config
