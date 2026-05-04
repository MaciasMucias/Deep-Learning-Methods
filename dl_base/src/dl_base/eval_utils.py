import csv
import re
import sys
from pathlib import Path

import numpy as np


def discover_seed_dirs(checkpoint_dir: Path, run_name: str) -> list[tuple[int, Path]]:
    """Return [(seed, path), ...] for all matching run directories, sorted by seed."""
    pattern = re.compile(rf"^{re.escape(run_name)}_seed\((\d+)\)$")
    matches = []
    for candidate in checkpoint_dir.iterdir():
        if not candidate.is_dir():
            continue
        m = pattern.match(candidate.name)
        if m:
            matches.append((int(m.group(1)), candidate))
    return sorted(matches, key=lambda x: x[0])


def aggregate_seed_results(
    seed_results: list[tuple],
) -> tuple[list[float], list[float], list[str]]:
    """
    Consume a list of (seed, loss, acc, *extra, err_msg) tuples.
    Prints per-seed status lines and returns (losses, accuracies, notes).
    err_msg is the last element; None means success.
    """
    losses: list[float] = []
    accuracies: list[float] = []
    notes: list[str] = []
    for row in seed_results:
        seed, loss, acc = row[0], row[1], row[2]
        err = row[-1]
        if err is not None:
            print(f"    seed({seed})  [FAILED] {err}", file=sys.stderr)
            notes.append(f"seed({seed}): FAILED")
        else:
            print(f"    seed({seed})  loss={loss:.4f}  acc={acc:.4f}")
            losses.append(loss)
            accuracies.append(acc)
            notes.append(f"seed({seed}): ok")
    return losses, accuracies, notes


def make_result_row(
    config_path: Path,
    model_name: str,
    run_name: str,
    losses: list[float],
    accuracies: list[float],
    notes: list[str],
    **extra_fields,
) -> dict:
    """Build the standard result row dict, with optional extra fields appended."""
    row = {
        "config": str(config_path),
        "model": model_name,
        "run_name": run_name,
        "n_seeds": len(accuracies),
        "mean_accuracy": round(float(np.mean(accuracies)), 4),
        "std_accuracy": round(float(np.std(accuracies)), 4),
        "mean_loss": round(float(np.mean(losses)), 4),
        "std_loss": round(float(np.std(losses)), 4),
        "seeds_status": " | ".join(notes),
    }
    row.update(extra_fields)
    return row


BASE_CSV_FIELDNAMES = [
    "config",
    "model",
    "run_name",
    "n_seeds",
    "mean_accuracy",
    "std_accuracy",
    "mean_loss",
    "std_loss",
    "seeds_status",
]


def write_results_csv(
    output_path: Path, rows: list[dict], fieldnames: list[str] = BASE_CSV_FIELDNAMES
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
