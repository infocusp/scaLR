from typing import Tuple, Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from torch.nn.functional import pad

from scalr.feature.scoring import ScoringBase
from scalr.nn.dataloader import build_dataloader
from scalr.nn.model import CustomShapModel
from scalr.utils import data_utils


class ShapScorer(ScoringBase):
    """This scorer is SHAP based. It can be used for any model."""

    def __init__(self,
                 early_stop: dict,
                 dataloader: dict,
                 dirpath: str = ".",
                 device: str = 'cpu',
                 top_n_genes: int = 100,
                 background_tensor: int = 200,
                 *args,
                 **kwargs):
        self.early_stop_config = early_stop
        self.dirpath = dirpath
        self.device = device
        self.top_n_genes = top_n_genes
        self.background_tensor = background_tensor
        self.dataloader_config = dataloader

    def generate_scores(self, model: nn.Module,
                        train_data: Union[AnnData, AnnCollection],
                        val_data: Union[AnnData, AnnCollection], target: str,
                        *args, **kwargs) -> np.ndarray:
        """Return the weights of model as score"""

        val_dl, _ = build_dataloader(self.dataloader_config, val_data, target,
                                     mappings)

        abs_mean_shap_values, _ = self.get_top_n_genes_weights(
            model,
            train_data,
            val_dl,
        )

        return abs_mean_shap_values.T

    def get_top_n_genes_weights(
        self,
        model: nn.Module,
        train_data: Union[AnnData, AnnCollection],
        test_dl: Union[AnnData, AnnCollection],
        batch_onehotencoder: OneHotEncoder = None,
    ) -> None:
        """
        Function to get top n genes of each class and its weights.

        Args:
            model: trained model to extract weights from
            train_data: train data.
            test_dl: test dataloader that used for shap values.

        Returns:
            class wise top n genes, genes * class weights matrix
        """

        model.to(self.device)
        shap_model = CustomShapModel(model)

        random_background_data = data_utils.get_random_samples(
            train_data,
            self.background_tensor,
            self.device,
            batch_onehotencoder,
        )

        padding = self.dataloader_config['params']['padding']
        if padding and random_background_data.shape[1] < padding:
            # Add padding(features) to data.
            random_background_data = pad(
                random_background_data,
                (0, padding - random_background_data.shape[1]), 'constant', 0.0)

        explainer = shap.DeepExplainer(shap_model, random_background_data)

        abs_prev_top_genes_batch_wise = {}
        count_patience = 0
        total_samples = 0

        for batch_id, batch in enumerate(test_dl):
            total_samples += batch[0].shape[0]

            batch_shap_values = explainer.shap_values(batch[0].to(self.device))

            abs_sum_shap_values = np.abs(batch_shap_values).sum(axis=0)
            # calcluating 2 mean with abs values and non-abs values.
            # Non-abs values required for heatmap.
            sum_shap_values = batch_shap_values.sum(axis=0)
            if batch_id >= 1:
                abs_sum_shap_values = np.sum(
                    [abs_sum_shap_values, abs_prev_batches_sum_shap_values],
                    axis=0)
                sum_shap_values = np.sum(
                    [sum_shap_values, prev_batches_sum_shap_values], axis=0)

            abs_mean_shap_values = abs_sum_shap_values / total_samples

            # Handle batch correction. Remove batch features from analysis.
            if batch_onehotencoder:
                abs_mean_shap_values = abs_mean_shap_values[:-len(
                    batch_onehotencoder.categories_[0]), :]

            abs_genes_class_shap_df = pd.DataFrame(
                abs_mean_shap_values[:len(test_dl.dataset.var_names)],
                index=test_dl.dataset.var_names)

            abs_prev_batches_sum_shap_values = abs_sum_shap_values
            prev_batches_sum_shap_values = sum_shap_values

            early_stop, abs_prev_top_genes_batch_wise = self._is_shap_early_stop(
                batch_id, abs_genes_class_shap_df,
                abs_prev_top_genes_batch_wise, self.top_n_genes,
                self.early_stop_config['threshold'])

            count_patience = count_patience + 1 if early_stop else 0

            if count_patience == self.early_stop_config['patience']:
                print(f"Early stopping at batch: {batch_id}")
                break

        mean_shap_values = sum_shap_values / total_samples

        # Handle batch correction. Remove batch features from analysis.
        if batch_onehotencoder:
            mean_shap_values = mean_shap_values[:-len(batch_onehotencoder.
                                                      categories_[0]), :]

        return abs_mean_shap_values, mean_shap_values

    def save_data():
        genes_class_shap_df = DataFrame(mean_shap_values,
                                        index=test_dl.dataset.var_names,
                                        columns=classes)

        abs_genes_class_shap_df = DataFrame(abs_mean_shap_values,
                                            index=test_dl.dataset.var_names,
                                            columns=classes)

        abs_genes_class_shap_df.T.to_csv(
            path.join(dirpath, "genes_class_weights.csv"))

        genes_class_shap_df.T.to_csv(
            path.join(dirpath, "raw_genes_class_weights.csv"))

        # Extract only top N genes
        class_top_genes = {
            class_label: genes[:top_n]
            for class_label, genes in abs_prev_top_genes_batch_wise.items()
        }

        return class_top_genes, genes_class_shap_df

    def _is_shap_early_stop(
        self,
        batch_id: int,
        genes_class_shap_df: pd.DataFrame,
        prev_top_genes_batch_wise: dict,
        top_n_genes: int,
        threshold: int,
    ) -> Tuple[bool, dict]:
        """Function to check whether previous and current batches' common genes are
            are greater than or equal to the threshold and return top genes
            batch wise.

        Args:
            batch_id: Current batch number.
            genes_class_shap_df: label/class wise genes shap values(mean across samples).
            prev_top_genes_batch_wise: dict where prev batch's per labels top genes are stored.
            top_n_genes: Number of top genes check.
            threshold: early stop if common genes is higher than this.

        Returns:
            early stop value, top genes batch wise.
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
        """Class method to get default params"""
        return {
            "top_n_genes": 100,
            "background_tensor": 200,
            "early_stop": {
                "patience": 5,
                "threshold": 95,
            }
        }
