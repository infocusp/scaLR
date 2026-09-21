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
import numpy as np
import torch

import scalr
from scalr.calibration import OpenSetThresholds
from scalr.calibration import predictive_entropy
from scalr.calibration import TemperatureScaler
from scalr.calibration import top1_top2_margin
from scalr.doublet import flag_possible_doublets
from scalr.genes import align_genes
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
    def predict(
        self,
        adata: AnnData,
        device: str = 'auto',
        batch_size: int = 4096,
        open_set: bool = True,
        top_k: int = 5,
        min_feature_overlap: float = 0.1,
        level: str = 'fine',
        cluster_refinement: Optional[str] = None,
        flag_doublets: bool = False,
    ) -> PredictionResult:
        """Run gene-aligned, calibrated inference on an AnnData object.

        Args:
            adata: Query AnnData object. Does not need to share the model's
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

        Returns:
            A `PredictionResult` aligned to `adata.obs_names`.
        """
        if level == 'broad' and not self.taxonomy:
            raise ValueError(
                'level="broad" requires a taxonomy; this model has none. '
                'Pass `taxonomy` when training/constructing the model.')
        report = validate(adata,
                          model_features=self.features,
                          min_feature_overlap=min_feature_overlap)
        report.raise_if_unusable()

        aligned, gene_report = align_genes(adata, self.features,
                                           min_feature_overlap)
        if not gene_report.is_usable:
            raise ValueError(
                f'Gene coverage too low for reliable prediction:\n{gene_report}'
            )

        resolved_device = _select_device(device)
        self.model.to(resolved_device)
        self.model.eval()

        n = aligned.shape[0]
        all_logits = []
        with torch.no_grad():
            for start in range(0, n, batch_size):
                chunk = aligned[start:start + batch_size].X
                if not isinstance(chunk, np.ndarray):
                    chunk = chunk.toarray()
                x = torch.as_tensor(chunk,
                                    dtype=torch.float32).to(resolved_device)
                out = self.model(x)['cls_output']
                all_logits.append(out.cpu())

        logits = torch.cat(all_logits, dim=0)
        probabilities = self.calibrator.calibrate_probabilities(logits.numpy())

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
            obs_names=list(adata.obs_names),
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
