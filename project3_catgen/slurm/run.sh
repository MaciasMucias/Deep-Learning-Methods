#!/usr/bin/env bash

#SBATCH --job-name=catgen-dcgan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=24:00:00
#SBATCH --partition=student
#SBATCH --account=stud-2526-l-03
#SBATCH --output=project3_catgen/slurm/logs/output_%j.out
#SBATCH --error=project3_catgen/slurm/logs/error_%j.err
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

CONFIG="project3_catgen/configs/dcgan/baseline.yaml"
SEED="42"
ATTEMPT="${ATTEMPT:-1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-23h30m}"
WANDB_MODE="${WANDB_MODE:-offline}"
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"

usage() {
  cat <<'EOF'
Usage:
  sbatch project3_catgen/slurm/run.sh --config CONFIG --seed SEED

Examples:
  sbatch project3_catgen/slurm/run.sh --config project3_catgen/configs/dcgan/baseline.yaml --seed 42
  sbatch project3_catgen/slurm/run.sh --config project3_catgen/configs/dcgan/baseline.yaml --seed 1
  sbatch project3_catgen/slurm/run.sh --config project3_catgen/configs/dcgan/baseline.yaml --seed 3
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --attempt)
      ATTEMPT="$2"
      shift 2
      ;;
    --max-attempts)
      MAX_ATTEMPTS="$2"
      shift 2
      ;;
    --train-timeout)
      TRAIN_TIMEOUT="$2"
      shift 2
      ;;
    --wandb-mode)
      WANDB_MODE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "$REPO_DIR"
mkdir -p project3_catgen/slurm/logs project3_catgen/runs

if [[ ! -f "$CONFIG" ]]; then
  echo "[error] config not found: $CONFIG" >&2
  exit 1
fi

echo "date: $(date)"
echo "node: ${SLURMD_NODENAME:-unknown}"
echo "job id: ${SLURM_JOB_ID:-local}"
echo "attempt: $ATTEMPT/$MAX_ATTEMPTS"
echo "config: $CONFIG"
echo "seed: $SEED"

command -v uv >/dev/null 2>&1 || {
  echo "[error] uv is not installed or not on PATH" >&2
  exit 1
}

uv sync --package project3-catgen --frozen

checkpoint_path() {
  uv run --package project3-catgen python - "$CONFIG" "$SEED" <<'PY'
import sys
from pathlib import Path
from project3_catgen.config import load_config

config = load_config(sys.argv[1])
seed = int(sys.argv[2])
print(Path(config.checkpoint_dir) / f"{config.run_name}_seed({seed})" / "last.pth")
PY
}

is_finished() {
  uv run --package project3-catgen python - "$CONFIG" "$SEED" <<'PY'
import sys
from pathlib import Path
import torch
from project3_catgen.config import load_config

config = load_config(sys.argv[1])
seed = int(sys.argv[2])
path = Path(config.checkpoint_dir) / f"{config.run_name}_seed({seed})" / "last.pth"

if not path.exists():
    print(f"checkpoint missing: {path}")
    sys.exit(1)

checkpoint = torch.load(path, map_location="cpu", weights_only=False)
done = int(checkpoint["epoch"]) + 1
target = int(config.training.num_epochs)
print(f"checkpoint: {path} ({done}/{target} epochs)")
sys.exit(0 if done >= target else 1)
PY
}

if is_finished; then
  echo "training already finished"
  exit 0
fi

RESUME_ARGS=()
CKPT="$(checkpoint_path)"
if [[ -f "$CKPT" ]]; then
  RESUME_ARGS=(--resume)
  echo "resuming from: $CKPT"
else
  echo "starting from scratch"
fi

TRAIN_CMD=(
  uv run --package project3-catgen python -m project3_catgen.experiments.train
  --config "$CONFIG"
  --seeds "$SEED"
  --wandb-mode "$WANDB_MODE"
  "${RESUME_ARGS[@]}"
)

echo "training command: ${TRAIN_CMD[*]}"

set +e
timeout --signal=TERM --kill-after=60s "$TRAIN_TIMEOUT" "${TRAIN_CMD[@]}"
RC=$?
set -e

echo "training exit code: $RC"

if is_finished; then
  echo "training finished"
  uv run --package project3-catgen python -m project3_catgen.experiments.interpolate \
    --config "$CONFIG" \
    --seed "$SEED" || true
  exit 0
fi

if [[ "$AUTO_RESUBMIT" != "1" ]]; then
  echo "[error] training incomplete and AUTO_RESUBMIT=$AUTO_RESUBMIT" >&2
  exit 1
fi

if (( ATTEMPT >= MAX_ATTEMPTS )); then
  echo "[error] training incomplete after $MAX_ATTEMPTS attempts" >&2
  exit 1
fi

case "$RC" in
  0|124|137|143)
    NEXT_ATTEMPT=$((ATTEMPT + 1))
    echo "resubmitting attempt $NEXT_ATTEMPT"
    sbatch \
      --job-name="${SLURM_JOB_NAME:-catgen-dcgan}" \
      project3_catgen/slurm/run.sh \
      --config "$CONFIG" \
      --seed "$SEED" \
      --attempt "$NEXT_ATTEMPT" \
      --max-attempts "$MAX_ATTEMPTS" \
      --train-timeout "$TRAIN_TIMEOUT" \
      --wandb-mode "$WANDB_MODE"
    ;;
  *)
    echo "[error] training failed with non-timeout code $RC; not resubmitting" >&2
    exit "$RC"
    ;;
esac
