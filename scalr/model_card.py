"""This file implements model card generation for provenance and reproducibility.

Every self-contained model artifact should document what it was trained on,
how it performs, and its known limitations, separately from its raw
`manifest.json` metadata — so a person deciding whether to trust/reuse the
model does not have to reverse-engineer the training run.
"""

_TEMPLATE = """# Model Card: {model_name}

## Summary
- **Task**: {task}
- **Model version**: {model_version}
- **scaLR version**: {scalr_version}
- **Species**: {species}
- **Random seed**: {random_seed}

## Training data
- **Labels column**: {labels_key}
- **Grouping column (leakage-safe split)**: {group_key}
- **Split ratio (train/val/test)**: {split_ratio}
- **Feature count**: {feature_count}
- **Classes ({n_classes})**: {class_list}

## Performance (held-out test split)
{metrics_section}

## Confidence & abstention
- Calibration: temperature scaling (T={temperature:.3f})
- Open-set thresholds: {open_set_thresholds}

## Known limitations
- Evaluated only on the training-distribution data described above; performance on
  out-of-distribution donors, tissues, or sequencing technologies is not validated.
- Open-set (`is_unknown`) thresholds are fit on a single validation split and are a
  heuristic screening signal, not a guarantee of correct abstention.
- Gene/feature coverage should be checked (`scalr.validate`) on every new query dataset.

## License
- Source code license: see repository LICENSE.
- This model artifact's weights/data license: not specified by this template;
  set explicitly by the model's author/distributor before sharing.
"""


def render_model_card(metadata: dict, metrics: dict) -> str:
    """Render a Markdown model card from an `AnnotationModel`'s manifest metadata and metrics.

    Args:
        metadata: The model's manifest dict (as written by `AnnotationModel.save`),
            containing keys such as `labels`, `model_version`, `labels_key`, etc.
        metrics: The model's recorded evaluation metrics dict (may be empty).

    Returns:
        Markdown text suitable for writing to the artifact's `README.md`.
    """
    class_names = metadata.get('labels', [])

    metrics_section = 'No held-out metrics recorded.'
    if metrics:
        lines = []
        if 'macro_f1' in metrics:
            lines.append(f"- macro-F1: {metrics['macro_f1']:.4f}")
        if 'weighted_f1' in metrics:
            lines.append(f"- weighted-F1: {metrics['weighted_f1']:.4f}")
        if 'balanced_accuracy' in metrics:
            lines.append(
                f"- balanced accuracy: {metrics['balanced_accuracy']:.4f}")
        if 'n_test_cells' in metrics:
            lines.append(f"- test cells: {metrics['n_test_cells']:,}")
        if lines:
            metrics_section = '\n'.join(lines)

    return _TEMPLATE.format(
        model_name=metadata.get('model_name') or metadata.get('labels_key') or
        'unnamed model',
        task=metadata.get('task', 'cell_type_annotation'),
        model_version=metadata.get('model_version', 'unknown'),
        scalr_version=metadata.get('scalr_version', 'unknown'),
        species=metadata.get('species') or 'not specified',
        random_seed=metadata.get('random_seed', 'unknown'),
        labels_key=metadata.get('labels_key', 'unknown'),
        group_key=metadata.get('group_key') or
        'none (no leakage-safe grouping used)',
        split_ratio=metadata.get('split_ratio', 'unknown'),
        feature_count=metadata.get('feature_count', 'unknown'),
        n_classes=len(class_names),
        class_list=', '.join(class_names) if class_names else 'unknown',
        metrics_section=metrics_section,
        temperature=metadata.get('temperature', 1.0),
        open_set_thresholds=metadata.get('open_set_thresholds', {}),
    )
