# Model Card: cell_type

## Summary
- **Task**: cell_type_annotation
- **Model version**: 1.0.0
- **scaLR version**: 2.0.0.dev0
- **Species**: not specified
- **Random seed**: 42

## Training data
- **Labels column**: cell_type
- **Grouping column (leakage-safe split)**: donor_id
- **Split ratio (train/val/test)**: [0.7, 0.1, 0.2]
- **Feature count**: 30695
- **Classes (10)**: CD16-negative, CD56-bright natural killer cell, human, CD16-positive, CD56-dim natural killer cell, human, classical monocyte, conventional dendritic cell, granulocyte, intermediate monocyte, natural killer cell, non-classical monocyte, plasmacytoid dendritic cell, platelet

## Performance (held-out test split)
- macro-F1: 0.6885
- weighted-F1: 0.9281
- balanced accuracy: 0.6822
- test cells: 23,492

## Confidence & abstention
- Calibration: temperature scaling (T=2.546)
- Open-set thresholds: {'min_confidence': 0.8577296197414398, 'max_entropy': 0.20046073645353296, 'min_margin': 0.7288531720638275}

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
