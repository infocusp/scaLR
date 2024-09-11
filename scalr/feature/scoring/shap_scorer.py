"""This file is an implementation of SHAP scorer."""

from typing import Tuple, Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import OneHotEncoder
import torch
from torch import nn

from scalr import utils
from scalr.feature.scoring import ScoringBase
from scalr.nn.dataloader import build_dataloader
from scalr.nn.model import CustomShapModel


class ShapScorer(ScoringBase):
    """Class for SHAP scorer. It can be used for any model."""

    def __init__(self,
                 early_stop: dict,
                 dataloader: dict,
                 device: str = 'cpu',
                 top_n_genes: int = 100,
                 background_tensor: int = 200,
                 samples_abs_mean: bool = True,
                 logger: str = 'EventLogger',
                 *args,
                 **kwargs):
        """Initialize class with SHAP arguments.

        Args:
            early_stop: Contains early stopping-related configuration.
            dataloader: Dataloader related config.
            device: Where data is processed/loaded.
            top_n_genes: Top N genes for each class/label.
            background_tensor: Number of training data used for SHAP explainer.
            samples_abs_mean: Apply abs before taking the mean across samples.
        """

        self.early_stop_config = early_stop
        self.device = device
        self.top_n_genes = top_n_genes
        self.background_tensor = background_tensor
        self.dataloader_config = dataloader
        self.samples_abs_mean = samples_abs_mean

        self.logger = getattr(utils, logger)('SHAP analysis')

    def generate_scores(self, model: nn.Module,
                        train_data: Union[AnnData, AnnCollection],
                        val_data: Union[AnnData, AnnCollection], target: str,
                        mappings: dict, *args, **kwargs) -> np.ndarray:
        """This function returns the weights of the model as a score.

        Args:
            model: Trained model that is used for SHAP.
            train_data: Data that is used as reference data for SHAP.
            val_data: On which SHAP will generate the score.
            mappings: Contains target-related mappings.

        Returns:
            class * genes abs weights matrix.
        """

        shap_values = self.get_top_n_genes_weights(model, train_data, val_data,
                                                   target, mappings)

        return shap_values

    def get_top_n_genes_weights(
            self, model: nn.Module, train_data: Union[AnnData, AnnCollection],
            test_data: Union[AnnData, AnnCollection], target: str,
            mappings: dict) -> Tuple[np.ndarray, np.ndarray]:
        """ A function to get top n genes of each class and its weights.

        Args:
            model: Trained model to extract weights from.
            train_data: Train data.
            test_data: Test data that is used for SHAP values.
            target: Target name.
            mappings: Contains target-related mappings.

        Returns:
            (class * genes abs weights matrix, class * genes weights matrix).
        """

        if isinstance(self.logger, utils.EventLogger):
            self.logger.heading2("Genes analysis using SHAP.")

        model.to(self.device)
        shap_model = CustomShapModel(model)

        random_indices = np.random.randint(0, train_data.shape[0],
                                           self.background_tensor)
        train_dl, _ = build_dataloader(self.dataloader_config,
                                       train_data[random_indices], target,
                                       mappings)
        random_background_data = torch.cat([batch[0] for batch in train_dl])

        self.logger.info(
            f"Selected random background data: {random_background_data.shape}")

        test_dl, _ = build_dataloader(self.dataloader_config, test_data, target,
                                      mappings)

        explainer = shap.DeepExplainer(shap_model,
                                       random_background_data.to(self.device))

        prev_top_genes_batch_wise = {}
        count_patience = 0
        total_samples = 0

        for batch_id, batch in enumerate(test_dl):
            self.logger.info(f"Running on batch: {batch_id}")
            total_samples += batch[0].shape[0]

            batch_shap_values = explainer.shap_values(batch[0].to(self.device))
            if self.samples_abs_mean:
                sum_shap_values = np.abs(batch_shap_values).sum(axis=0)
            else:
                # Calcluating 2 mean with abs values and non-abs values.
                # Non-abs values required for heatmap.
                sum_shap_values = batch_shap_values.sum(axis=0)

            if batch_id >= 1:
                sum_shap_values = np.sum(
                    [sum_shap_values, prev_batches_sum_shap_values], axis=0)

            mean_shap_values = sum_shap_values / total_samples

            genes_class_shap_df = pd.DataFrame(
                mean_shap_values[:len(test_dl.dataset.var_names)],
                index=test_dl.dataset.var_names)

            prev_batches_sum_shap_values = sum_shap_values

            early_stop, prev_top_genes_batch_wise = self._is_shap_early_stop(
                batch_id, genes_class_shap_df, prev_top_genes_batch_wise,
                self.top_n_genes, self.early_stop_config['threshold'])

            count_patience = count_patience + 1 if early_stop else 0

            if count_patience == self.early_stop_config['patience']:
                self.logger.info(f"Early stopping at batch: {batch_id}")
                break

        return mean_shap_values.T

    def _is_shap_early_stop(
        self,
        batch_id: int,
        genes_class_shap_df: pd.DataFrame,
        prev_top_genes_batch_wise: dict,
        top_n_genes: int,
        threshold: int,
    ) -> Tuple[bool, dict]:
        """A function to check whether previous and current batches' common genes are
            are greater than or equal to the threshold and return top genes
            batch wise.

        Args:
            batch_id: Current batch number.
            genes_class_shap_df: label/class wise genes SHAP values(mean across samples).
            prev_top_genes_batch_wise: Dictionary where prev batches per labels top genes are stored.
            top_n_genes: Number of top genes check.
            threshold: early stop if common genes are higher than this.

        Returns:
            Early stop value, top genes batch wise.
        """

        early_stop = True
        top_genes_batch_wise = {}
        classes = genes_class_shap_df.columns
        for label in classes:
            top_genes_batch_wise[label] = genes_class_shap_df[
                label].sort_values(ascending=False)[:top_n_genes].index

            # Start checking after first batch.
            if batch_id >= 1:
                num_common_genes = len(
                    set(top_genes_batch_wise[label]).intersection(
                        set(prev_top_genes_batch_wise[label])))
                # If commnon genes are less than 90 early stop will be false.
                if num_common_genes < threshold:
                    early_stop = False
            else:
                early_stop = False

        return early_stop, top_genes_batch_wise

    @classmethod
    def get_default_params(cls) -> dict:
        """Class method to get default params."""
        return {
            "top_n_genes": 100,
            "background_tensor": 200,
            "samples_abs_mean": True,
            "early_stop": {
                "patience": 5,
                "threshold": 95
            },
            "dataloader": {
                "name": "SimpleDataLoader",
                "params": {
                    "batch_size": 5000,
                    "padding": 5000
                }
            }
        }
