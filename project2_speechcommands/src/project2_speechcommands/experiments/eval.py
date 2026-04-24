#!/usr/bin/env python3
"""
Evaluate all configs across all discovered seeds, write aggregated CSV, and save confusion matrices.

Usage:
    uv run python -m project2_speechcommands.experiments.eval \
        --configs "project2_speechcommands/configs/cnn_baseline/*.yaml" \
        --checkpoint best \
        --output results.csv \
        --cm-dir project2_speechcommands/runs/confusion_matrices
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
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
from project2_speechcommands.config import load_config, ExperimentConfig
from project2_speechcommands.data import (
    get_dataloaders,
    CORE_COMMANDS,
    SILENCE_LABEL,
    UNKNOWN_LABEL,
    PRELIM_CLASS_NAMES,
)
from project2_speechcommands.experiments.utils import build_model

CLASS_NAMES = CORE_COMMANDS + ["silence", "unknown"]


class TwoStageClassifier:
    """
    Stage 1: prelim_model → 3-class: 0=known, 1=unknown, 2=silence
    Stage 2: known → main_model (12-class, labels 0-11)
             unknown → UNKNOWN_LABEL (11) directly
             silence → SILENCE_LABEL (10) directly

    main_model has 12 outputs so it can still predict silence/unknown as a
    safety net if the prelim makes an error.
    """

    PRELIM_KNOWN, PRELIM_UNKNOWN, PRELIM_SILENCE = 0, 1, 2

    def __init__(
        self,
        main_model: nn.Module,
        prelim_model: nn.Module,
        device: torch.device,
    ) -> None:
        self.main_model = main_model
        self.prelim_model = prelim_model
        self.device = device

    def predict_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.to(self.device)
        preds = torch.full((inputs.size(0),), UNKNOWN_LABEL, dtype=torch.long)

        with torch.no_grad():
            prelim_preds = torch.argmax(self.prelim_model(inputs), dim=1)

        preds[(prelim_preds == self.PRELIM_SILENCE).cpu()] = SILENCE_LABEL

        known_mask = prelim_preds == self.PRELIM_KNOWN
        if known_mask.any():
            with torch.no_grad():
                main_preds = torch.argmax(
                    self.main_model(inputs[known_mask]), dim=1
                ).cpu()
            preds[known_mask.cpu()] = main_preds

        return preds


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch eval: autodiscover seeds, aggregate mean±std, write CSV + confusion matrices."
    )
    parser.add_argument("--configs", type=str, required=True)
    parser.add_argument(
        "--checkpoint", type=str, choices=["best", "last"], default="best"
    )
    parser.add_argument("--output", type=str, default="eval_results.csv")
    parser.add_argument(
        "--cm-dir",
        type=str,
        default="project2_speechcommands/runs/confusion_matrices",
        help="Directory to save confusion matrix PNGs",
    )
    parser.add_argument(
        "--binary-checkpoint",
        type=str,
        default=None,
        help="Base checkpoint directory for the prelim model (e.g. project2_speechcommands/runs). "
        "Enables two-stage inference; requires --prelim-config.",
    )
    parser.add_argument(
        "--prelim-config",
        type=str,
        default=None,
        help="Path to prelim model config YAML. Required when --binary-checkpoint is provided.",
    )
    return parser.parse_args()


_test_loader_cache: dict[tuple[Path, int, str], DataLoader] = {}


def get_test_loader(config: ExperimentConfig) -> DataLoader:
    strategy = config.balance.strategy
    key = (config.data_root, config.training.batch_size, strategy)
    if key not in _test_loader_cache:
        print(f"  [cache miss] loading test set ...", flush=True)
        _, _, loader = get_dataloaders(
            config.data_root,
            config.audio,
            config.balance,
            config.training.batch_size,
            num_workers=0,
            test_mode=True,
            remap_prelim=(strategy == "prelim"),
        )
        _test_loader_cache[key] = loader
    else:
        print(f"  [cache hit]  reusing test set", flush=True)
    return _test_loader_cache[key]



def eval_config(
    config: ExperimentConfig,
    seed_dirs: list[tuple[int, Path]],
    checkpoint: str,
    cm_dir: Path,
    binary_checkpoint_path: str | None,
    prelim_config: ExperimentConfig | None,
) -> tuple[
    list[tuple[int, float | None, float | None, dict | None, str | None]], np.ndarray
]:
    """
    Evaluate all seeds for one config.
    Returns:
      - list of (seed, loss, accuracy, per_class_acc_dict, error_msg)
      - aggregate confusion matrix (sum across seeds)
    """
    use_two_stage = binary_checkpoint_path is not None and prelim_config is not None
    test_loader = get_test_loader(config)
    device = get_device()
    results = []
    n_out_classes = 12 if use_two_stage else config.num_classes
    agg_cm = np.zeros((n_out_classes, n_out_classes), dtype=int)

    for seed, _ in seed_dirs:
        try:
            set_seed(seed)
            checkpoint_dir = config.checkpoint_dir / f"{config.run_name}_seed({seed})"

            model = build_model(config)
            model.to(device)
            optimizer = AdamW(
                model.parameters(),
                lr=config.training.lr,
                weight_decay=config.training.weight_decay,
            )
            criterion = nn.CrossEntropyLoss()

            trainer = Trainer(model, optimizer, criterion, device, checkpoint_dir)
            trainer.load_checkpoint(checkpoint, restore_rng=False)

            all_preds, all_targets = [], []

            if use_two_stage:
                prelim_ckpt_dir = (
                    Path(binary_checkpoint_path)
                    / f"{prelim_config.run_name}_seed({seed})"
                )
                prelim_model = build_model(prelim_config)
                prelim_model.to(device)
                prelim_optimizer = AdamW(
                    prelim_model.parameters(),
                    lr=prelim_config.training.lr,
                    weight_decay=prelim_config.training.weight_decay,
                )
                prelim_trainer = Trainer(
                    prelim_model, prelim_optimizer, criterion, device, prelim_ckpt_dir
                )
                prelim_trainer.load_checkpoint(checkpoint)

                classifier = TwoStageClassifier(model, prelim_model, device)
                model.eval()
                prelim_model.eval()
                for inputs, targets in test_loader:
                    batch_preds = classifier.predict_batch(inputs)
                    all_preds.extend(batch_preds.tolist())
                    all_targets.extend(targets.tolist())

                correct = sum(p == t for p, t in zip(all_preds, all_targets))
                acc = correct / len(all_targets)
                loss = float("nan")
            else:
                model.eval()
                with torch.no_grad():
                    for inputs, targets in test_loader:
                        inputs = inputs.to(device)
                        logits = model(inputs)
                        preds = torch.argmax(logits, dim=1).cpu()
                        all_preds.extend(preds.tolist())
                        all_targets.extend(targets.tolist())
                loss, acc = trainer.test(test_loader)

            cm = confusion_matrix(
                all_targets, all_preds, labels=list(range(n_out_classes))
            )
            agg_cm += cm
            out_class_names = CLASS_NAMES if n_out_classes == 12 else PRELIM_CLASS_NAMES
            per_class_acc = {
                out_class_names[i]: round(cm[i, i] / cm[i].sum(), 4)
                if cm[i].sum() > 0
                else 0.0
                for i in range(n_out_classes)
            }

            results.append((seed, loss, acc, per_class_acc, None))

        except Exception as e:
            results.append((seed, None, None, None, str(e)))

    return results, agg_cm


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    run_name: str,
    cm_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cmap="Blues",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{run_name} — Confusion Matrix (aggregate)")
    fig.tight_layout()
    cm_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(cm_dir / f"cm_{run_name}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = get_args()
    cm_dir = Path(args.cm_dir)

    if args.binary_checkpoint and not args.prelim_config:
        print(
            "[error] --binary-checkpoint requires --prelim-config to specify the prelim model architecture.",
            file=sys.stderr,
        )
        sys.exit(1)

    prelim_config: ExperimentConfig | None = None
    if args.prelim_config:
        try:
            prelim_config = load_config(args.prelim_config)
        except Exception as e:
            print(f"[error] Failed to load prelim config: {e}", file=sys.stderr)
            sys.exit(1)

    config_paths = sorted(Path(".").glob(args.configs.lstrip("./")))
    if not config_paths:
        print(f"[error] No config files matched: {args.configs}", file=sys.stderr)
        sys.exit(1)

    use_two_stage = args.binary_checkpoint is not None and prelim_config is not None
    print(f"Found {len(config_paths)} config(s). Checkpoint: {args.checkpoint}\n")

    rows: list[dict] = []

    for config_path in config_paths:
        try:
            config = load_config(config_path)
        except Exception as e:
            print(f"[skip] {config_path}: failed to load — {e}", file=sys.stderr)
            continue

        seed_dirs = discover_seed_dirs(config.checkpoint_dir, config.run_name)
        if not seed_dirs:
            print(f"[skip] {config.run_name}: no seed directories found")
            continue

        print(
            f"{config.run_name}  ({len(seed_dirs)} seed(s): {[s for s, _ in seed_dirs]})"
        )

        seed_results, agg_cm = eval_config(
            config,
            seed_dirs,
            args.checkpoint,
            cm_dir,
            args.binary_checkpoint,
            prelim_config,
        )

        losses, accuracies, notes = aggregate_seed_results(seed_results)

        if not accuracies:
            print(f"  → no successful runs, skipping\n")
            continue

        if use_two_stage:
            cm_names = CLASS_NAMES
        elif config.balance.strategy == "prelim":
            cm_names = PRELIM_CLASS_NAMES
        else:
            cm_names = CLASS_NAMES[: config.num_classes]
        save_confusion_matrix(agg_cm, cm_names, config.run_name, cm_dir)

        # Use per_class from last successful seed for the CSV
        last_per_class = next(
            (pc for _, _, _, pc, err in reversed(seed_results) if err is None), {}
        )

        rows.append(make_result_row(
            config_path, config.model_name, config.run_name, losses, accuracies, notes,
            per_class_accuracy=json.dumps(last_per_class),
        ))
        print(
            f"  → mean_acc={rows[-1]['mean_accuracy']:.4f} ± {rows[-1]['std_accuracy']:.4f}\n"
        )

    if not rows:
        print("[error] No results to write.", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    write_results_csv(output_path, rows, BASE_CSV_FIELDNAMES + ["per_class_accuracy"])
    print(f"Results written to: {output_path}  ({len(rows)} rows)")
    print(f"Confusion matrices saved to: {cm_dir}/")


if __name__ == "__main__":
    main()
