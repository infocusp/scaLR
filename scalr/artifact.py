"""This file implements the self-contained, versioned model artifact format.

A scaLR model artifact is a directory containing everything needed to
reproduce inference without the original training config:

    model/
    ├── manifest.json       # format/version, task, species, labels, hashes
    ├── model.pt            # model weights (scalr.nn.model state dict)
    ├── model_config.json   # class name + params to rebuild the architecture
    ├── label_mapping.json  # id2label / label2id
    ├── features.json       # ordered gene/feature list the model expects
    ├── preprocessing.json  # fit-time preprocessing contract
    ├── calibration.json    # temperature scaling + open-set thresholds
    └── metrics.json        # evaluation metrics recorded at training time
"""

from dataclasses import asdict
import json
import os
from os import path
from typing import Optional, Union

from anndata import AnnData
from anndata.experimental import AnnCollection
import numpy as np
import torch

import scalr
from scalr.calibration import OpenSetThresholds
from scalr.calibration import predictive_entropy
from scalr.calibration import TemperatureScaler
from scalr.calibration import top1_top2_margin
from scalr.doublet import flag_possible_doublets
from scalr.explain import attribute_cells
from scalr.explain import attribute_class
from scalr.explain import CellExplanation
from scalr.genes import align_genes
from scalr.genes import GeneAlignmentReport
from scalr.hierarchy import aggregate_to_broad
from scalr.hierarchy import validate_taxonomy
from scalr.model_card import render_model_card
from scalr.nn.model import build_model
from scalr.refinement import refine_with_clusters
from scalr.result import PredictionResult
from scalr.utils import read_data
from scalr.utils import write_data
from scalr.validation import validate

FORMAT_VERSION = 1


def _select_device(device: str) -> str:
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


class AnnotationModel:
    """A trained, self-contained scaLR annotation model.

    Wraps a `scalr.nn.model` network together with its label mapping, feature
    list, preprocessing contract and confidence calibrator, so it can be
    saved/loaded and used for inference without any external config.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_config: dict,
        class_names: list[str],
        features: list[str],
        preprocessing: Optional[dict] = None,
        calibrator: Optional[TemperatureScaler] = None,
        open_set_thresholds: Optional[OpenSetThresholds] = None,
        metrics: Optional[dict] = None,
        metadata: Optional[dict] = None,
        taxonomy: Optional[dict[str, str]] = None,
    ):
        self.model = model
        self.model_config = model_config
        self.class_names = class_names
        self.features = features
        self.preprocessing = preprocessing or {}
        self.calibrator = calibrator or TemperatureScaler(temperature=1.0)
        self.open_set_thresholds = open_set_thresholds or OpenSetThresholds()
        self.metrics = metrics or {}
        self.metadata = metadata or {}
        if taxonomy is not None:
            validate_taxonomy(class_names, taxonomy)
        self.taxonomy = taxonomy

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _score_aligned(self, aligned: AnnData, batch_size: int,
                       device: str) -> np.ndarray:
        """Run the network over an already gene-aligned AnnData in batches
        and return calibrated probabilities. Never densifies more than one
        `batch_size` chunk of `aligned.X` at a time."""
        n = aligned.shape[0]
        all_logits = []
        with torch.no_grad():
            for start in range(0, n, batch_size):
                chunk = aligned[start:start + batch_size].X
                if not isinstance(chunk, np.ndarray):
                    chunk = chunk.toarray()
                x = torch.as_tensor(chunk, dtype=torch.float32).to(device)
                out = self.model(x)['cls_output']
                all_logits.append(out.cpu())
        logits = torch.cat(all_logits, dim=0)
        return self.calibrator.calibrate_probabilities(logits.numpy())

    def _predict_streaming(
        self,
        source: Union[AnnData, AnnCollection],
        chunk_size: int,
        min_feature_overlap: float,
        batch_size: int,
        device: str,
    ) -> tuple[list[str], np.ndarray, GeneAlignmentReport]:
        """Score `source` `chunk_size` cells at a time, only ever
        gene-aligning one chunk in memory at once, so peak alignment/scoring
        memory does not grow with total cell count.

        For a single backed `.h5ad` file, `source` is first fully loaded
        (backed row-slicing of a single file cannot be done lazily here); to
        avoid ever holding the whole dataset in memory, pass a directory of
        chunked `.h5ad` files (read as an `AnnCollection`), which is sliced
        and loaded one chunk at a time.
        """
        if isinstance(source, AnnData) and source.isbacked:
            source = source.to_memory(copy=True)

        n_total = source.shape[0]
        obs_names: list[str] = []
        probability_chunks = []
        gene_report = None

        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            chunk = source[start:end]
            if not isinstance(chunk, AnnData):
                chunk = chunk.to_adata()
            if hasattr(chunk, 'to_memory'):
                chunk = chunk.to_memory(copy=True)

            if gene_report is None:
                report = validate(chunk,
                                  model_features=self.features,
                                  min_feature_overlap=min_feature_overlap)
                report.raise_if_unusable()

            aligned_chunk, chunk_gene_report = align_genes(
                chunk, self.features, min_feature_overlap)
            if not chunk_gene_report.is_usable:
                raise ValueError('Gene coverage too low for reliable '
                                 f'prediction:\n{chunk_gene_report}')
            if gene_report is None:
                gene_report = chunk_gene_report

            obs_names.extend(list(chunk.obs_names))
            probability_chunks.append(
                self._score_aligned(aligned_chunk, batch_size, device))

        return obs_names, np.concatenate(probability_chunks,
                                         axis=0), gene_report

    def predict(
        self,
        adata: Union[AnnData, AnnCollection, str],
        device: str = 'auto',
        batch_size: int = 4096,
        open_set: bool = True,
        top_k: int = 5,
        min_feature_overlap: float = 0.1,
        level: str = 'fine',
        cluster_refinement: Optional[str] = None,
        flag_doublets: bool = False,
        streaming: bool = False,
        chunk_size: int = 20000,
    ) -> PredictionResult:
        """Run gene-aligned, calibrated inference on an AnnData object.

        Args:
            adata: Query AnnData/AnnCollection object, or a path to an
                `.h5ad` file or directory of chunked `.h5ad` files (read via
                `scalr.utils.read_data`). Does not need to share the model's
                gene set or gene order; alignment is performed automatically.
            device: 'auto', 'cpu' or 'cuda'.
            batch_size: Number of cells scored per forward pass.
            open_set: When True, cells failing the model's confidence/entropy/
                margin thresholds are flagged `is_unknown=True`.
            top_k: Number of top classes to report per cell.
            min_feature_overlap: Minimum required gene-overlap fraction;
                raises if coverage falls below this.
            level: 'fine' (default, the model's native classes) or 'broad' to
                aggregate predictions to broad classes using `self.taxonomy`
                (raises if the model has no taxonomy).
            cluster_refinement: When given, relabel cells to their cluster's
                confidence-weighted majority class. Either an `adata.obs`
                column name, or 'auto' to detect a 'leiden'/'louvain'/
                'cluster'/'clusters' column. `None` (default) disables this;
                the raw, unrefined labels are always kept in
                `result.metadata['raw_labels']` when refinement is applied.
            flag_doublets: When True, screen for cells whose top-two class
                probabilities are both substantial and close together, and
                set `result.is_possible_doublet` — a heuristic screening
                signal, not a validated doublet call.
            streaming: When True, or implicitly when `adata` is a path, gene-
                align and score the data `chunk_size` cells at a time instead
                of materializing the whole aligned matrix at once, bounding
                peak alignment/scoring memory to roughly
                `chunk_size x n_features` regardless of total dataset size.
                For out-of-core reads too (never loading the full raw dataset
                into memory), pass a directory of chunked `.h5ad` files
                rather than a single large `.h5ad` file.
            chunk_size: Cells scored per chunk when `streaming`/path input is
                used.

        Returns:
            A `PredictionResult` aligned to `adata.obs_names`.
        """
        if level == 'broad' and not self.taxonomy:
            raise ValueError(
                'level="broad" requires a taxonomy; this model has none. '
                'Pass `taxonomy` when training/constructing the model.')

        if isinstance(adata, str):
            adata = read_data(adata, backed='r')
            streaming = True

        resolved_device = _select_device(device)
        self.model.to(resolved_device)
        self.model.eval()

        if streaming or not isinstance(adata, AnnData):
            obs_names, probabilities, gene_report = self._predict_streaming(
                adata, chunk_size, min_feature_overlap, batch_size,
                resolved_device)
        else:
            report = validate(adata,
                              model_features=self.features,
                              min_feature_overlap=min_feature_overlap)
            report.raise_if_unusable()

            aligned, gene_report = align_genes(adata, self.features,
                                               min_feature_overlap)
            if not gene_report.is_usable:
                raise ValueError('Gene coverage too low for reliable '
                                 f'prediction:\n{gene_report}')
            obs_names = list(adata.obs_names)
            probabilities = self._score_aligned(aligned, batch_size,
                                                resolved_device)

        n = len(obs_names)
        top1_ids = probabilities.argmax(axis=1)
        labels = [self.class_names[i] for i in top1_ids]
        confidence = probabilities.max(axis=1)
        entropy = predictive_entropy(probabilities)
        margin = top1_top2_margin(probabilities)

        if open_set:
            is_unknown = self.open_set_thresholds.flag_unknown(
                confidence, entropy, margin)
        else:
            is_unknown = np.zeros(n, dtype=bool)

        k = min(top_k, len(self.class_names))
        top_k_out = []
        top_k_idx = np.argsort(-probabilities, axis=1)[:, :k]
        for row_idx, row in enumerate(top_k_idx):
            top_k_out.append([(self.class_names[c],
                               float(probabilities[row_idx, c])) for c in row])

        is_possible_doublet = None
        if flag_doublets:
            is_possible_doublet = flag_possible_doublets(
                probabilities, entropy, margin)

        result = PredictionResult(
            obs_names=obs_names,
            labels=labels,
            probabilities=probabilities,
            class_names=self.class_names,
            confidence=confidence,
            entropy=entropy,
            margin=margin,
            is_unknown=is_unknown,
            top_k=top_k_out,
            is_possible_doublet=is_possible_doublet,
            metadata={
                'model_version': self.metadata.get('model_version'),
                'scalr_version': self.metadata.get('scalr_version'),
                'temperature': self.calibrator.temperature,
                'gene_coverage': gene_report.coverage,
                'missing_features': gene_report.missing_features,
                'open_set_thresholds': self.open_set_thresholds.to_dict(),
                'device': resolved_device,
            },
        )

        if level == 'broad':
            result = aggregate_to_broad(result, self.taxonomy)

        if cluster_refinement is not None:
            result = refine_with_clusters(result,
                                          adata,
                                          cluster_key=cluster_refinement)

        return result

    # ------------------------------------------------------------------ #
    # Explainability
    # ------------------------------------------------------------------ #
    def explain(
        self,
        adata: AnnData,
        indices: list[int],
        top_k: int = 20,
        device: str = 'auto',
        min_feature_overlap: float = 0.1,
    ) -> list[CellExplanation]:
        """Explain individual cell predictions via Grad x Input gene attribution.

        For each requested cell, reports the model's prediction and the genes
        whose expression most supported (positive attribution) or contradicted
        (negative attribution) that prediction. This is a first-order
        sensitivity explanation, not a validated biological one — see
        `scalr.explain` for the method.

        Args:
            adata: Query AnnData object; gene-aligned automatically.
            indices: Positional row indices into `adata` to explain.
            top_k: Number of supporting and contradictory genes to report per cell.
            device: 'auto', 'cpu' or 'cuda'.
            min_feature_overlap: Minimum required gene-overlap fraction.

        Returns:
            One `CellExplanation` per entry in `indices`, in the same order.
        """
        resolved_device = _select_device(device)
        self.model.to(resolved_device)
        self.model.eval()

        aligned, gene_report = align_genes(adata, self.features,
                                           min_feature_overlap)
        if not gene_report.is_usable:
            raise ValueError('Gene coverage too low for reliable '
                             f'explanation:\n{gene_report}')

        subset = aligned[indices]
        chunk = subset.X
        if not isinstance(chunk, np.ndarray):
            chunk = chunk.toarray()
        x = torch.as_tensor(chunk, dtype=torch.float32).to(resolved_device)

        with torch.no_grad():
            logits = self.model(x)['cls_output']
        probabilities = self.calibrator.calibrate_probabilities(
            logits.cpu().numpy())
        predicted_ids = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        obs_names = [str(adata.obs_names[i]) for i in indices]

        return attribute_cells(self.model,
                               x,
                               self.features,
                               self.class_names,
                               predicted_ids,
                               confidence,
                               obs_names,
                               top_k=top_k)

    def explain_class(
        self,
        class_name: str,
        top_k: int = 50,
        device: str = 'auto',
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """Explain a class in general, via gradient at a zero-expression baseline.

        Returns:
            `(supporting_genes, contradictory_genes)`, each a list of
            `(gene, score)` pairs. See `scalr.explain.attribute_class`.
        """
        resolved_device = _select_device(device)
        self.model.to(resolved_device)
        self.model.eval()
        return attribute_class(self.model,
                               self.features,
                               self.class_names,
                               class_name,
                               top_k=top_k,
                               device=resolved_device)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, dirpath: str) -> None:
        """Save this model as a self-contained artifact directory."""
        os.makedirs(dirpath, exist_ok=True)

        self.model.save_weights(path.join(dirpath, 'model.pt'))
        write_data(self.model_config, path.join(dirpath, 'model_config.json'))
        write_data(
            {
                'label2id': {
                    c: i for i, c in enumerate(self.class_names)
                },
                'id2label': {
                    i: c for i, c in enumerate(self.class_names)
                },
            }, path.join(dirpath, 'label_mapping.json'))
        write_data({'features': self.features},
                   path.join(dirpath, 'features.json'))
        write_data(self.preprocessing, path.join(dirpath, 'preprocessing.json'))
        write_data(
            {
                'calibration': self.calibrator.to_dict(),
                'open_set_thresholds': self.open_set_thresholds.to_dict(),
            }, path.join(dirpath, 'calibration.json'))
        write_data(self.metrics, path.join(dirpath, 'metrics.json'))

        if self.taxonomy is not None:
            write_data({'taxonomy': self.taxonomy},
                       path.join(dirpath, 'taxonomy.json'))

        manifest = {
            'format_version': FORMAT_VERSION,
            'scalr_version': getattr(scalr, '__version__', 'unknown'),
            'task': 'cell_type_annotation',
            'feature_count': len(self.features),
            'labels': self.class_names,
            **{
                k: v for k, v in self.metadata.items() if k not in ('scalr_version',)
            },
            'temperature': self.calibrator.temperature,
            'open_set_thresholds': self.open_set_thresholds.to_dict(),
            'has_taxonomy': self.taxonomy is not None,
        }
        write_data(manifest, path.join(dirpath, 'manifest.json'))

        with open(path.join(dirpath, 'README.md'), 'w') as fh:
            fh.write(render_model_card(manifest, self.metrics))

    @classmethod
    def load(cls, dirpath: str) -> 'AnnotationModel':
        """Load a self-contained model artifact directory."""
        manifest = read_data(path.join(dirpath, 'manifest.json'))
        if manifest.get('format_version', 1) > FORMAT_VERSION:
            raise ValueError(
                f'Model artifact format_version={manifest["format_version"]} is '
                f'newer than the format this scaLR version supports '
                f'({FORMAT_VERSION}). Please upgrade scaLR.')

        model_config = read_data(path.join(dirpath, 'model_config.json'))
        model, model_config = build_model(model_config)
        model.load_weights(path.join(dirpath, 'model.pt'))

        label_mapping = read_data(path.join(dirpath, 'label_mapping.json'))
        id2label = label_mapping['id2label']
        class_names = [id2label[str(i)] for i in range(len(id2label))]

        features = read_data(path.join(dirpath, 'features.json'))['features']

        preprocessing = {}
        if path.exists(path.join(dirpath, 'preprocessing.json')):
            preprocessing = read_data(path.join(dirpath, 'preprocessing.json'))

        calibrator = TemperatureScaler()
        open_set_thresholds = OpenSetThresholds()
        calib_path = path.join(dirpath, 'calibration.json')
        if path.exists(calib_path):
            calib = read_data(calib_path)
            if calib.get('calibration'):
                calibrator = TemperatureScaler.from_dict(calib['calibration'])
            if calib.get('open_set_thresholds'):
                open_set_thresholds = OpenSetThresholds.from_dict(
                    calib['open_set_thresholds'])

        metrics = {}
        metrics_path = path.join(dirpath, 'metrics.json')
        if path.exists(metrics_path):
            metrics = read_data(metrics_path)

        taxonomy = None
        taxonomy_path = path.join(dirpath, 'taxonomy.json')
        if path.exists(taxonomy_path):
            taxonomy = read_data(taxonomy_path)['taxonomy']

        return cls(
            model=model,
            model_config=model_config,
            taxonomy=taxonomy,
            class_names=class_names,
            features=features,
            preprocessing=preprocessing,
            calibrator=calibrator,
            open_set_thresholds=open_set_thresholds,
            metrics=metrics,
            metadata=manifest,
        )


def load_model(dirpath: str) -> AnnotationModel:
    """Load a self-contained scaLR model artifact from `dirpath`."""
    return AnnotationModel.load(dirpath)
