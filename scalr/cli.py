"""This file implements the `scalr` command-line interface for the simple public API.

    scalr validate --input data.h5ad [--labels-key cell_type] [--group-key donor_id]
    scalr train --input train.h5ad --labels-key cell_type [--group-key donor_id] --output models/pbmc_v1
    scalr annotate --input data.h5ad --model models/pbmc_v1 --output annotated.h5ad
    scalr evaluate --input test.h5ad --model models/pbmc_v1 --labels-key cell_type
    scalr models list
    scalr models info <name>

This CLI wraps the in-memory simple API (`scalr.train`/`annotate`/`validate`);
the configuration-driven, chunked/streaming pipeline remains available via
`python pipeline.py --config ...` for large-scale/advanced runs.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import scalr
from scalr.metrics import compute_classification_metrics
from scalr.utils import read_data
from scalr.utils import write_data


def _cmd_validate(args) -> int:
    adata = read_data(args.input, backed=None)
    report = scalr.validate(adata,
                            labels_key=args.labels_key,
                            group_key=args.group_key,
                            min_feature_overlap=args.min_feature_overlap)
    print(report)
    return 0 if report.is_usable else 1


def _cmd_train(args) -> int:
    adata = read_data(args.input, backed=None)
    features = None
    if args.features_file:
        with open(args.features_file, encoding='utf-8') as handle:
            features = json.load(handle)
    taxonomy = None
    if args.taxonomy_file:
        with open(args.taxonomy_file, encoding='utf-8') as handle:
            taxonomy = json.load(handle)
    model = scalr.train(adata,
                        labels_key=args.labels_key,
                        group_key=args.group_key,
                        features=features,
                        hidden_layers=tuple(args.hidden_layers),
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        device=args.device,
                        split_ratio=tuple(args.split_ratio),
                        class_balanced_loss=not args.no_class_balanced_loss,
                        open_set=not args.no_open_set,
                        taxonomy=taxonomy,
                        seed=args.seed,
                        verbose=not args.quiet)
    model.save(args.output)
    print(f'Model saved to {args.output}')
    if model.metrics:
        print(f"macro-F1: {model.metrics.get('macro_f1', float('nan')):.4f}  "
              f"balanced accuracy: "
              f"{model.metrics.get('balanced_accuracy', float('nan')):.4f}")
    return 0


def _cmd_annotate(args) -> int:
    adata = read_data(args.input, backed=None)
    result = scalr.annotate(adata,
                            model=args.model,
                            device=args.device,
                            open_set=not args.no_open_set,
                            min_feature_overlap=args.min_feature_overlap,
                            flag_doublets=args.flag_doublets)
    result.write_to_adata(adata)
    write_data(adata, args.output)

    n_unknown = int(result.is_unknown.sum())
    print(f'Annotated {len(result)} cells ({n_unknown} flagged unknown). '
          f'Written to {args.output}')
    return 0


def _cmd_evaluate(args) -> int:
    adata = read_data(args.input, backed=None)
    model = scalr.load_model(args.model)
    result = model.predict(adata, device=args.device)

    true_labels = adata.obs[args.labels_key].astype(str).values
    label2id = {c: i for i, c in enumerate(model.class_names)}
    unseen_labels = sorted(set(true_labels) - set(label2id))
    if unseen_labels:
        print(
            f'WARNING: {len(unseen_labels)} true label(s) not known to the '
            f'model will be excluded from evaluation: {unseen_labels}',
            file=sys.stderr)

    keep_mask = [label in label2id for label in true_labels]
    y_true = [
        label2id[label] for label, keep in zip(true_labels, keep_mask) if keep
    ]
    y_pred = [
        label2id[pred] for pred, keep in zip(result.labels, keep_mask) if keep
    ]

    metrics = compute_classification_metrics(y_true, y_pred, model.class_names)
    print(metrics)
    print()
    print(metrics.per_class)
    return 0


def _cmd_models_list(args) -> int:
    names = scalr.models.list()
    if not names:
        print('No models registered locally.')
        return 0
    for name in names:
        print(name)
    return 0


def _cmd_models_info(args) -> int:
    manifest = scalr.models.info(args.name)
    print(json.dumps(manifest, indent=2))
    return 0


def _cmd_analyze(args) -> int:
    """Run configured downstream analyses through the pipeline runner."""
    pipeline_path = Path(__file__).resolve().parent.parent / 'pipeline.py'
    command = [sys.executable, str(pipeline_path), '--config', args.config]
    if args.log:
        command.append('--log')
    if args.level:
        command.extend(['--level', args.level])
    if args.logpath:
        command.extend(['--logpath', args.logpath])
    if args.memoryprofiler:
        command.append('--memoryprofiler')
    return subprocess.run(command, check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    """Build the `scalr` CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog='scalr', description='scaLR 2.0 command-line interface.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    p = subparsers.add_parser(
        'validate', help='Validate an AnnData file for use with scaLR.')
    p.add_argument('--input', required=True, help='Path to an .h5ad file.')
    p.add_argument('--labels-key', default=None)
    p.add_argument('--group-key', default=None)
    p.add_argument('--min-feature-overlap', type=float, default=0.1)
    p.set_defaults(func=_cmd_validate)

    p = subparsers.add_parser('train', help='Train a scaLR annotation model.')
    p.add_argument('--input',
                   required=True,
                   help='Path to a training .h5ad file.')
    p.add_argument('--labels-key', required=True)
    p.add_argument('--group-key', default=None)
    p.add_argument('--features-file', default=None,
                   help='JSON file containing the ordered input gene list.')
    p.add_argument('--hidden-layers', nargs='+', type=int, default=[256, 64],
                   help='Hidden layer sizes. Default: 256 64.')
    p.add_argument('--output',
                   required=True,
                   help='Directory to save the model artifact to.')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3,
                   help='Adam learning rate. Default: 0.001.')
    p.add_argument('--device', default='auto')
    p.add_argument('--split-ratio', nargs=3, type=float, default=[0.7, 0.1, 0.2],
                   metavar=('TRAIN', 'VAL', 'TEST'))
    p.add_argument('--no-class-balanced-loss', action='store_true',
                   help='Disable inverse-frequency class weighting.')
    p.add_argument('--no-open-set', action='store_true',
                   help='Disable unknown-cell abstention thresholds.')
    p.add_argument('--taxonomy-file', default=None,
                   help='JSON file mapping fine labels to broad labels.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--quiet', action='store_true')
    p.set_defaults(func=_cmd_train)

    p = subparsers.add_parser(
        'annotate', help='Annotate cells using a trained scaLR model.')
    p.add_argument('--input', required=True, help='Path to a query .h5ad file.')
    p.add_argument(
        '--model',
        required=True,
        help='Model artifact directory, or a locally registered model name.')
    p.add_argument('--output',
                   required=True,
                   help='Path to write the annotated .h5ad to.')
    p.add_argument('--device', default='auto')
    p.add_argument('--no-open-set',
                   action='store_true',
                   help='Disable open-set abstention flagging.')
    p.add_argument('--flag-doublets',
                   action='store_true',
                   help='Screen for possible mixed/doublet cells.')
    p.add_argument('--min-feature-overlap', type=float, default=0.1)
    p.set_defaults(func=_cmd_annotate)

    p = subparsers.add_parser(
        'evaluate', help='Evaluate a trained model against labeled test data.')
    p.add_argument('--input',
                   required=True,
                   help='Path to a labeled test .h5ad file.')
    p.add_argument('--model', required=True)
    p.add_argument('--labels-key', required=True)
    p.add_argument('--device', default='auto')
    p.set_defaults(func=_cmd_evaluate)

    p_models = subparsers.add_parser('models',
                                     help='Manage locally registered models.')
    models_sub = p_models.add_subparsers(dest='models_command', required=True)
    p_list = models_sub.add_parser('list',
                                   help='List locally registered models.')
    p_list.set_defaults(func=_cmd_models_list)
    p_info = models_sub.add_parser('info',
                                   help="Show a registered model's manifest.")
    p_info.add_argument('name')
    p_info.set_defaults(func=_cmd_models_info)

    p = subparsers.add_parser(
        'analyze',
        help='Run downstream analyses configured in a YAML pipeline config.')
    p.add_argument('--config', required=True, help='Path to config.yaml.')
    p.add_argument('--log', action='store_true', help='Save experiment logs.')
    p.add_argument('--level', default=None, help='Logging level, e.g. INFO.')
    p.add_argument('--logpath', default=None, help='Path to the log file.')
    p.add_argument('--memoryprofiler', action='store_true',
                   help='Record peak memory usage.')
    p.set_defaults(func=_cmd_analyze)

    return parser


def main(argv: list = None) -> int:
    """Entry point for the `scalr` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
