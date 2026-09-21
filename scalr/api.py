"""This file implements the scaLR 2.0 simple, inference-first public API.

    import scalr

    model = scalr.train(adata, labels_key="cell_type", group_key="donor_id")
    model.save("models/pbmc_v1")

    model = scalr.load_model("models/pbmc_v1")
    result = model.predict(adata)

    result = scalr.annotate(adata, model="models/pbmc_v1")
"""

from typing import Optional, Union

from anndata import AnnData
import numpy as np
import torch
from torch import nn

import scalr
from scalr.artifact import AnnotationModel
from scalr.artifact import load_model as _load_model
from scalr.calibration import OpenSetThresholds
from scalr.calibration import TemperatureScaler
from scalr.leakage import check_group_leakage
from scalr.leakage import detect_likely_grouping_column
from scalr.leakage import group_safe_split
from scalr.metrics import check_class_imbalance
from scalr.metrics import compute_class_weights
from scalr.metrics import compute_classification_metrics
from scalr.nn.model import build_model
from scalr.result import PredictionResult
from scalr.utils import FlowLogger
from scalr.utils import set_seed
from scalr.validation import detect_normalization_state
from scalr.validation import validate


def _select_device(device: str) -> str:
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def _dense_batch(adata: AnnData, indices: np.ndarray, start: int, end: int):
    chunk = adata[indices[start:end]].X
    if not isinstance(chunk, np.ndarray):
        chunk = chunk.toarray()
    return np.asarray(chunk, dtype=np.float32)


def _iterate_batches(adata: AnnData, indices: np.ndarray, label_ids: np.ndarray,
                     batch_size: int, shuffle: bool = False):
    order = indices.copy()
    if shuffle:
        np.random.shuffle(order)
    for start in range(0, len(order), batch_size):
        batch_idx = order[start:start + batch_size]
        x = _dense_batch(adata, batch_idx, 0, len(batch_idx))
        y = label_ids[batch_idx]
        yield torch.as_tensor(x), torch.as_tensor(y, dtype=torch.long)


def train(
    adata: AnnData,
    labels_key: str,
    group_key: Optional[str] = None,
    features: Optional[list[str]] = None,
    hidden_layers: tuple = (256, 64),
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = 'auto',
    split_ratio: tuple = (0.7, 0.1, 0.2),
    class_balanced_loss: bool = True,
    open_set: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> AnnotationModel:
    """Train a self-contained scaLR annotation model from an AnnData object.

    Performs leakage-safe splitting (grouped by `group_key` when provided),
    class-weighted training, temperature-scaling calibration, and evaluation
    with macro/weighted-F1 and balanced accuracy on a held-out test split.

    Args:
        adata: Training AnnData; `adata.X` should already be normalized
            (the scaLR convention). `adata.obs[labels_key]` holds cell-type labels.
        labels_key: Column in `adata.obs` with classification labels.
        group_key: Column in `adata.obs` identifying donor/patient/sample.
            When provided, splitting keeps every group in a single split,
            preventing train/test leakage. Strongly recommended for reference
            datasets with donor/patient structure.
        features: Ordered gene list to use as model input. Defaults to
            `adata.var_names`.
        hidden_layers: Sizes of hidden layers of the classification network.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Adam learning rate.
        device: 'auto', 'cpu' or 'cuda'.
        split_ratio: (train, val, test) ratio for the internal split.
        class_balanced_loss: When True, use inverse-frequency class weights in
            the training loss, improving rare-cell performance.
        open_set: When True, fit open-set (unknown-cell) abstention thresholds
            from the validation split's confidence/entropy/margin distribution.
        seed: Random seed for reproducibility.
        verbose: Log progress via scaLR's FlowLogger.

    Returns:
        A trained, calibrated, self-contained `AnnotationModel`.
    """
    logger = FlowLogger('scalr.train')
    set_seed(seed)

    report = validate(adata, labels_key=labels_key, group_key=group_key)
    report.raise_if_unusable()
    if verbose:
        for w in report.warnings:
            logger.warning(w)

    if group_key is None:
        detected = detect_likely_grouping_column(adata.obs)
        if detected:
            logger.warning(
                f'No group_key given, but column(s) {detected} look like a '
                'donor/patient/sample identifier. Consider passing '
                'group_key to avoid train/test leakage.')

    imbalance = check_class_imbalance(adata.obs[labels_key])
    if imbalance.small_classes and verbose:
        logger.warning(str(imbalance))

    features = list(features) if features is not None else list(
        adata.var_names)
    class_names = sorted(adata.obs[labels_key].astype(str).unique().tolist())
    label2id = {c: i for i, c in enumerate(class_names)}
    label_ids = adata.obs[labels_key].astype(str).map(label2id).values

    split = group_safe_split(adata.obs, labels_key, group_key, split_ratio,
                             seed)
    if group_key is not None:
        leakage = check_group_leakage(
            {name: adata.obs.iloc[idx] for name, idx in split.items()},
            group_key)
        if leakage.has_leakage:
            logger.warning(str(leakage))

    resolved_device = _select_device(device)
    model_config = {
        'name': 'SequentialModel',
        'params': {
            'layers': [len(features), *hidden_layers,
                      len(class_names)],
        },
    }
    model, model_config = build_model(model_config)
    model.to(resolved_device)

    weights = None
    if class_balanced_loss:
        class_weights = compute_class_weights(label_ids[split['train']],
                                              len(class_names))
        weights = torch.as_tensor(class_weights, dtype=torch.float32).to(
            resolved_device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, y in _iterate_batches(adata,
                                    split['train'],
                                    label_ids,
                                    batch_size,
                                    shuffle=True):
            x, y = x.to(resolved_device), y.to(resolved_device)
            optimizer.zero_grad()
            out = model(x)['cls_output']
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if verbose:
            logger.info(
                f'epoch={epoch + 1}/{epochs} train_loss={epoch_loss / max(n_batches, 1):.4f}'
            )

    # --- Calibration on the validation split -------------------------------
    model.eval()
    val_logits, val_labels = [], []
    with torch.no_grad():
        for x, y in _iterate_batches(adata, split['val'], label_ids,
                                    batch_size):
            x = x.to(resolved_device)
            out = model(x)['cls_output']
            val_logits.append(out.cpu())
            val_labels.append(y)
    val_logits = torch.cat(val_logits, dim=0)
    val_labels = torch.cat(val_labels, dim=0)

    calibrator = TemperatureScaler()
    if len(val_labels) > 0:
        calibrator.fit(val_logits, val_labels)

    val_probs = calibrator.calibrate_probabilities(val_logits.numpy())
    open_set_thresholds = OpenSetThresholds()
    if open_set and len(val_labels) > 0:
        from scalr.calibration import predictive_entropy
        from scalr.calibration import top1_top2_margin
        confidence = val_probs.max(axis=1)
        entropy = predictive_entropy(val_probs)
        margin = top1_top2_margin(val_probs)
        open_set_thresholds = OpenSetThresholds(
            min_confidence=float(np.quantile(confidence, 0.05)),
            max_entropy=float(np.quantile(entropy, 0.95)),
            min_margin=float(np.quantile(margin, 0.05)),
        )

    # --- Evaluation on the held-out test split ------------------------------
    test_logits, test_labels = [], []
    with torch.no_grad():
        for x, y in _iterate_batches(adata, split['test'], label_ids,
                                    batch_size):
            x = x.to(resolved_device)
            out = model(x)['cls_output']
            test_logits.append(out.cpu())
            test_labels.append(y)

    metrics = {}
    if test_logits:
        test_probs = calibrator.calibrate_probabilities(
            torch.cat(test_logits, dim=0).numpy())
        test_preds = test_probs.argmax(axis=1)
        test_labels_np = torch.cat(test_labels, dim=0).numpy()
        class_metrics = compute_classification_metrics(
            test_labels_np, test_preds, class_names)
        metrics = {
            'macro_f1': class_metrics.macro_f1,
            'weighted_f1': class_metrics.weighted_f1,
            'balanced_accuracy': class_metrics.balanced_accuracy,
            'per_class': class_metrics.per_class.reset_index().to_dict(
                orient='records'),
            'n_test_cells': int(len(test_labels_np)),
        }
        if verbose:
            logger.info(str(class_metrics))

    norm_state = detect_normalization_state(adata)
    preprocessing = {
        'normalization': norm_state,
        'feature_count': len(features),
    }

    annotation_model = AnnotationModel(
        model=model,
        model_config=model_config,
        class_names=class_names,
        features=features,
        preprocessing=preprocessing,
        calibrator=calibrator,
        open_set_thresholds=open_set_thresholds,
        metrics=metrics,
        metadata={
            'scalr_version': getattr(scalr, '__version__', 'unknown'),
            'model_version': '1.0.0',
            'species': None,
            'labels_key': labels_key,
            'group_key': group_key,
            'random_seed': seed,
            'split_ratio': list(split_ratio),
        },
    )
    return annotation_model


def load_model(dirpath: str) -> AnnotationModel:
    """Load a self-contained scaLR model artifact from `dirpath`."""
    return _load_model(dirpath)


def annotate(
    adata: AnnData,
    model: Union[str, AnnotationModel],
    device: str = 'auto',
    preprocess: str = 'auto',
    open_set: bool = True,
    top_k: int = 5,
    min_feature_overlap: float = 0.1,
) -> PredictionResult:
    """Annotate cells in `adata` using a trained scaLR model.

    Args:
        adata: Query AnnData object.
        model: Either a path to a saved model artifact directory, or an
            already-loaded `AnnotationModel`.
        device: 'auto', 'cpu' or 'cuda'.
        preprocess: 'auto' detects and reports the query's normalization state
            versus the model's training-time contract, warning (not silently
            transforming) on a mismatch. 'skip' disables this check.
        open_set: When True, low-confidence/high-entropy/low-margin cells are
            flagged `is_unknown=True` instead of forced into a class.
        top_k: Number of top classes to report per cell.
        min_feature_overlap: Minimum required gene-overlap fraction.

    Returns:
        A `PredictionResult` aligned to `adata.obs_names`.
    """
    if isinstance(model, str):
        model = load_model(model)

    if preprocess == 'auto':
        query_state = detect_normalization_state(adata)
        expected_state = model.preprocessing.get('normalization')
        if expected_state and query_state not in (expected_state, 'unknown'):
            FlowLogger('scalr.annotate').warning(
                f'Query data looks like "{query_state}" but the model was '
                f'trained on "{expected_state}" data. Predictions may be unreliable.'
            )
    elif preprocess != 'skip':
        raise NotImplementedError(
            f'preprocess="{preprocess}" is not supported yet; use "auto" or "skip".'
        )

    return model.predict(adata,
                         device=device,
                         open_set=open_set,
                         top_k=top_k,
                         min_feature_overlap=min_feature_overlap)
