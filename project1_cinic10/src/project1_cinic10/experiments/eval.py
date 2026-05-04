#!/usr/bin/env python3
"""
Evaluate all configs across all discovered seeds and write aggregated results to CSV.

Usage:
    uv run project1_cinic10/src/project1_cinic10/experiments/eval_all.py \
        --configs "project1_cinic10/configs/**/*.yaml" \
        --checkpoint best \
        --output results.csv
"""

import argparse
import sys
from pathlib import Path

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dl_base import get_device, set_seed, Trainer
from dl_base import (
    discover_seed_dirs,
    aggregate_seed_results,
    make_result_row,
    write_results_csv,
    BASE_CSV_FIELDNAMES,
)
from project1_cinic10.config import load_config, ExperimentConfig
from project1_cinic10.data import get_dataloaders
from project1_cinic10.models import MODEL_REGISTRY


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch eval: autodiscover seeds, aggregate mean±std, write CSV."
    )
    parser.add_argument(
        "--configs",
        type=str,
        required=True,
        help='Glob pattern for config files, e.g. "project1_cinic10/configs/**/*.yaml"',
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        choices=["best", "last"],
        default="best",
        help="Which checkpoint to load (default: best)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results.csv",
        help="Path for the output CSV (default: eval_results.csv)",
    )
    return parser.parse_args()


_test_loader_cache: dict[tuple[Path, int], DataLoader] = {}


def get_test_loader(config: ExperimentConfig) -> DataLoader:
    key = (config.data_root, config.training.batch_size)
    if key not in _test_loader_cache:
        print(
            f"  [cache miss] loading test set (batch_size={config.training.batch_size})...",
            flush=True,
        )
        _, _, loader = get_dataloaders(
            config.data_root,
            config.augmentation,
            config.training.batch_size,
            0,  # num_workers=0 — no worker processes needed or wanted
            test_mode=True,
        )
        _test_loader_cache[key] = loader
    else:
        print(
            f"  [cache hit]  reusing test set (batch_size={config.training.batch_size})",
            flush=True,
        )
    return _test_loader_cache[key]



def eval_config(
    config: ExperimentConfig,
    seed_dirs: list[tuple[int, Path]],
    checkpoint: str,
) -> list[tuple[int, float | None, float | None, str | None]]:
    """
    Evaluate all seeds for one config, reusing a cached test_loader.
    Returns [(seed, loss, accuracy, error_msg), ...] where error_msg is None on success.
    """
    test_loader = get_test_loader(config)
    device = get_device()
    results = []

    for seed, _ in seed_dirs:
        try:
            set_seed(seed)
            checkpoint_dir = config.checkpoint_dir / f"{config.run_name}_seed({seed})"

            model = MODEL_REGISTRY[config.model_name](dropout=config.training.dropout)
            model.to(device)
            optimizer = AdamW(
                model.parameters(),
                lr=config.training.lr,
                weight_decay=config.training.weight_decay,
            )
            criterion = nn.CrossEntropyLoss()

            trainer = Trainer(model, optimizer, criterion, device, checkpoint_dir)
            trainer.load_checkpoint(checkpoint)

            loss, acc = trainer.test(test_loader)
            results.append((seed, loss, acc, None))

        except Exception as e:
            results.append((seed, None, None, str(e)))

    return results


def main() -> None:
    args = get_args()

    config_paths = sorted(Path(".").glob(args.configs.lstrip("./")))
    if not config_paths:
        print(f"[error] No config files matched: {args.configs}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(config_paths)} config(s). Checkpoint: {args.checkpoint}\n")

    rows: list[dict] = []

    for config_path in config_paths:
        try:
            config = load_config(config_path)
        except Exception as e:
            print(f"[skip] {config_path}: failed to load config — {e}", file=sys.stderr)
            continue

        seed_dirs = discover_seed_dirs(config.checkpoint_dir, config.run_name)
        if not seed_dirs:
            print(
                f"[skip] {config.run_name}: no seed directories found in {config.checkpoint_dir}"
            )
            continue

        print(
            f"{config.run_name}  ({len(seed_dirs)} seed(s): {[s for s, _ in seed_dirs]})"
        )

        seed_results = eval_config(config, seed_dirs, args.checkpoint)

        losses, accuracies, notes = aggregate_seed_results(seed_results)

        if not accuracies:
            print(f"  → no successful runs, skipping row\n")
            continue

        rows.append(make_result_row(config_path, config.model_name, config.run_name, losses, accuracies, notes))
        print(
            f"  → mean_acc={rows[-1]['mean_accuracy']:.4f} ± {rows[-1]['std_accuracy']:.4f}\n"
        )

    if not rows:
        print("[error] No results to write.", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    write_results_csv(output_path, rows, BASE_CSV_FIELDNAMES)
    print(f"Results written to: {output_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
