import argparse
import subprocess
import sys
from pathlib import Path


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="offline",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    config_dir = Path(__file__).parent
    configs = sorted(config_dir.glob("*.yaml"))

    for config in configs:
        print(f"\nRunning {config.name}")
        command = [
            "uv",
            "run",
            "--package",
            "project3-catgen",
            "python",
            "-m",
            "project3_catgen.experiments.train",
            "--config",
            str(config),
            "--seeds",
            *map(str, args.seeds),
            "--wandb-mode",
            args.wandb_mode,
        ]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
