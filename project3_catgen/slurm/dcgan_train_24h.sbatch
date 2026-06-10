#!/usr/bin/env bash
#SBATCH --account=stud-2526-l-03
#SBATCH --partition=student
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
CONFIG="${CONFIG:-project3_catgen/configs/dcgan/baseline.yaml}"
SEED="${SEED:-42}"
ATTEMPT="${ATTEMPT:-1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-23h30m}"
WANDB_MODE="${WANDB_MODE:-offline}"
RUN_INTERPOLATION="${RUN_INTERPOLATION:-1}"
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
UV_PYTHON="${UV_PYTHON:-3.13}"
UV_SYNC_ARGS="${UV_SYNC_ARGS:---frozen}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-stud-2526-l-03}"
SBATCH_PARTITION="${SBATCH_PARTITION:-student}"
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
SBATCH_GPUS="${SBATCH_GPUS:-1}"
SBATCH_CPUS="${SBATCH_CPUS:-8}"
SBATCH_MEM="${SBATCH_MEM:-32G}"
SBATCH_JOB_NAME="${SBATCH_JOB_NAME:-${SLURM_JOB_NAME:-dcgan_project3}}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/project3_catgen/slurm/logs}"

die() {
  echo "[error] $*" >&2
  exit 1
}

find_python() {
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    die "python3 or python is required"
  fi
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  find_python
  echo "[setup] uv not found; installing it into the user site-packages"
  "$PYTHON_BIN" -m pip install --user uv
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv was installed but is still not on PATH"
}

sync_environment() {
  ensure_uv
  if [[ -n "$UV_PYTHON" ]]; then
    uv python install "$UV_PYTHON" || echo "[warn] uv could not install Python $UV_PYTHON; trying to use an existing interpreter"
    uv sync --package project3-catgen --python "$UV_PYTHON" $UV_SYNC_ARGS
  else
    uv sync --package project3-catgen $UV_SYNC_ARGS
  fi
}

last_checkpoint_path() {
  uv run --package project3-catgen python - "$CONFIG" "$SEED" <<'PY'
import sys
from pathlib import Path
from project3_catgen.config import load_config

config = load_config(sys.argv[1])
seed = int(sys.argv[2])
print(Path(config.checkpoint_dir) / f"{config.run_name}_seed({seed})" / "last.pth")
PY
}

checkpoint_status() {
  uv run --package project3-catgen python - "$CONFIG" "$SEED" <<'PY'
import sys
from pathlib import Path
import torch
from project3_catgen.config import load_config

config = load_config(sys.argv[1])
seed = int(sys.argv[2])
checkpoint_path = Path(config.checkpoint_dir) / f"{config.run_name}_seed({seed})" / "last.pth"
target_epochs = int(config.training.num_epochs)

if not checkpoint_path.exists():
    print(f"checkpoint=missing path={checkpoint_path} target_epochs={target_epochs}")
    sys.exit(2)

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
finished_epochs = int(checkpoint["epoch"]) + 1
print(f"checkpoint={checkpoint_path} finished_epochs={finished_epochs} target_epochs={target_epochs}")
sys.exit(0 if finished_epochs >= target_epochs else 1)
PY
}

run_interpolation() {
  if [[ "$RUN_INTERPOLATION" != "1" ]]; then
    return
  fi
  echo "[post] generating latent interpolation for seed=$SEED"
  uv run --package project3-catgen python -m project3_catgen.experiments.interpolate \
    --config "$CONFIG" \
    --seed "$SEED" || echo "[warn] interpolation failed; training checkpoint is still complete"
}

resubmit_self() {
  next_attempt=$((ATTEMPT + 1))
  if (( next_attempt > MAX_ATTEMPTS )); then
    die "max attempts reached ($MAX_ATTEMPTS); not resubmitting"
  fi

  mkdir -p "$LOG_DIR"
  echo "[slurm] resubmitting attempt $next_attempt/$MAX_ATTEMPTS"
  sbatch \
    --job-name="$SBATCH_JOB_NAME" \
    --account="$SBATCH_ACCOUNT" \
    --partition="$SBATCH_PARTITION" \
    --time="$SBATCH_TIME" \
    --cpus-per-task="$SBATCH_CPUS" \
    --mem="$SBATCH_MEM" \
    --gres="gpu:rtx6000:$SBATCH_GPUS" \
    --output="$LOG_DIR/%x_%j.out" \
    --error="$LOG_DIR/%x_%j.err" \
    --export=ALL,REPO_DIR="$REPO_DIR",CONFIG="$CONFIG",SEED="$SEED",ATTEMPT="$next_attempt",MAX_ATTEMPTS="$MAX_ATTEMPTS",TRAIN_TIMEOUT="$TRAIN_TIMEOUT",WANDB_MODE="$WANDB_MODE",RUN_INTERPOLATION="$RUN_INTERPOLATION",AUTO_RESUBMIT="$AUTO_RESUBMIT",UV_PYTHON="$UV_PYTHON",UV_SYNC_ARGS="$UV_SYNC_ARGS",SBATCH_ACCOUNT="$SBATCH_ACCOUNT",SBATCH_PARTITION="$SBATCH_PARTITION",SBATCH_TIME="$SBATCH_TIME",SBATCH_GPUS="$SBATCH_GPUS",SBATCH_CPUS="$SBATCH_CPUS",SBATCH_MEM="$SBATCH_MEM",LOG_DIR="$LOG_DIR",SBATCH_JOB_NAME="$SBATCH_JOB_NAME" \
    "$0"
}

echo "[job] id=${SLURM_JOB_ID:-local} host=$(hostname) attempt=$ATTEMPT/$MAX_ATTEMPTS"
echo "[job] repo=$REPO_DIR"
echo "[job] config=$CONFIG seed=$SEED wandb_mode=$WANDB_MODE"

cd "$REPO_DIR"
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1
export WANDB_MODE="$WANDB_MODE"
export WANDB_DIR="${WANDB_DIR:-$REPO_DIR/project3_catgen/runs/wandb}"
mkdir -p "$WANDB_DIR"

sync_environment

if checkpoint_status; then
  echo "[done] training already complete"
  run_interpolation
  exit 0
fi

resume_args=()
last_checkpoint="$(last_checkpoint_path)"
if [[ -f "$last_checkpoint" ]]; then
  echo "[resume] found $last_checkpoint"
  resume_args=(--resume)
else
  echo "[resume] no checkpoint found; starting from epoch 1"
fi

train_cmd=(
  uv run --package project3-catgen python -m project3_catgen.experiments.train
  --config "$CONFIG"
  --seeds "$SEED"
  --wandb-mode "$WANDB_MODE"
  "${resume_args[@]}"
)

echo "[train] timeout=$TRAIN_TIMEOUT command=${train_cmd[*]}"
set +e
if command -v timeout >/dev/null 2>&1; then
  timeout --kill-after=60s --signal=TERM "$TRAIN_TIMEOUT" "${train_cmd[@]}"
  train_rc=$?
else
  echo "[warn] timeout command not found; relying only on Slurm wall time"
  "${train_cmd[@]}"
  train_rc=$?
fi
set -e
echo "[train] exit_code=$train_rc"

if checkpoint_status; then
  echo "[done] training reached target epochs"
  run_interpolation
  exit 0
fi

case "$train_rc" in
  0|124|137|143)
    if [[ "$AUTO_RESUBMIT" == "1" ]]; then
      resubmit_self
      exit 0
    fi
    die "training incomplete and AUTO_RESUBMIT=$AUTO_RESUBMIT"
    ;;
  *)
    die "training failed with non-timeout exit code $train_rc; not resubmitting"
    ;;
esac
