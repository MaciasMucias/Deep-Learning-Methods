#!/usr/bin/env python3
"""
Train a speech command classification model.

Usage:
    uv run python -m project2_speechcommands.experiments.train \
        --config project2_speechcommands/configs/cnn_baseline/cnn_lr2_bs2.yaml \
        --seeds 42
"""

import argparse

from dl_base import get_device, count_parameters
from project2_speechcommands.config import load_config
from project2_speechcommands.experiments.utils import setup_experiment


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 42])
    parser.add_argument(
        "--resume", action="store_true", help="Resume from 'last' checkpoint"
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    config = load_config(args.config)

    for seed in args.seeds:
        print(f"Running {config.run_name} | seed={seed} | device={get_device().type}")
        trainer, train_loader, val_loader, _ = setup_experiment(config, seed)
        print(f"  Parameters: {count_parameters(trainer.model):,}")

        if args.resume:
            trainer.load_checkpoint("last")

        trainer.fit(
            train_loader,
            val_loader,
            config.training.num_epochs,
            config.project_name,
            config.run_name,
            f"{config.run_name}_seed{seed}",
        )


if __name__ == "__main__":
    main()
