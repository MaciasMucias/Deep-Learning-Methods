import subprocess
from pathlib import Path
import os

# ===== CONFIG =====
SEEDS = [1, 2, 42]
SCRIPT = "project2_speechcommands.experiments.train"
LOG_FILE = "completed_runs_cnn_baseline.txt"
# ==================


def load_completed(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()
    with open(log_path, "r") as f:
        lines = f.readlines()
    return {line.strip() for line in lines if line.strip()}


def append_completed(log_path: Path, name: str):
    with open(log_path, "a") as f:
        f.write(name + "\n")


def main():
    os.environ["WANDB_MODE"] = "offline"

    current_dir = Path(__file__).parent
    configs = sorted(current_dir.glob("*.yaml"))

    if not configs:
        print("No config files found in this folder!")
        return

    log_path = current_dir / LOG_FILE
    completed = load_completed(log_path)

    print(f"Working in: {current_dir}")
    print(f"Found {len(configs)} configs")
    print(f"Already completed: {len(completed)}\n")

    for cfg in configs:
        if cfg.name in completed:
            print(f"-> Skipping (already done): {cfg.name}")
            continue

        print(f"\nRunning: {cfg.name}\n")

        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            SCRIPT,
            "--config",
            str(cfg),
            "--seeds",
            *map(str, SEEDS),
        ]

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"\nFAILED: {cfg.name}")
            print("Stopping execution (fix error and resume later)")
            break
        else:
            print(f"\nDONE: {cfg.name}")
            append_completed(log_path, cfg.name)

    print("\nFinished!!!")


if __name__ == "__main__":
    main()
