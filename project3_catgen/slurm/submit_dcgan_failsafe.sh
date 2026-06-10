#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash project3_catgen/slurm/submit_dcgan_failsafe.sh [options]

Options:
  --config PATH          DCGAN YAML config. Default: project3_catgen/configs/dcgan/baseline.yaml
  --seed N              Submit one seed. Default: 42
  --seeds N [N ...]     Submit one independent Slurm job per seed
  --account NAME        Slurm account. Default: stud-2526-l-03
  --partition NAME      Slurm partition. Default: student
  --time HH:MM:SS       Slurm wall time. Default: 24:00:00
  --train-timeout DUR   Stop Python before Slurm kills the job. Default: 23h30m
  --gpus N              Number of RTX6000 GPUs. Default: 1
  --cpus N              CPUs per task. Default: 8
  --mem SIZE            Memory. Default: 32G
  --wandb-mode MODE     online, offline, or disabled. Default: offline
  --max-attempts N      Max auto-resubmissions per seed. Default: 30
  --no-interpolation    Do not run latent interpolation after final completion
  --test-only           Ask Slurm for scheduling estimate instead of submitting
  -h, --help            Show this help
EOF
}

die() {
  echo "[error] $*" >&2
  exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_dir="$(cd "$script_dir/../.." && pwd -P)"

CONFIG="project3_catgen/configs/dcgan/baseline.yaml"
SEEDS=("42")
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-stud-2526-l-03}"
SBATCH_PARTITION="${SBATCH_PARTITION:-student}"
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-23h30m}"
SBATCH_GPUS="${SBATCH_GPUS:-1}"
SBATCH_CPUS="${SBATCH_CPUS:-8}"
SBATCH_MEM="${SBATCH_MEM:-32G}"
WANDB_MODE="${WANDB_MODE:-offline}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
RUN_INTERPOLATION="${RUN_INTERPOLATION:-1}"
TEST_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --seed)
      SEEDS=("$2")
      shift 2
      ;;
    --seeds)
      shift
      SEEDS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        SEEDS+=("$1")
        shift
      done
      ;;
    --account)
      SBATCH_ACCOUNT="$2"
      shift 2
      ;;
    --partition)
      SBATCH_PARTITION="$2"
      shift 2
      ;;
    --time)
      SBATCH_TIME="$2"
      shift 2
      ;;
    --train-timeout)
      TRAIN_TIMEOUT="$2"
      shift 2
      ;;
    --gpus)
      SBATCH_GPUS="$2"
      shift 2
      ;;
    --cpus)
      SBATCH_CPUS="$2"
      shift 2
      ;;
    --mem)
      SBATCH_MEM="$2"
      shift 2
      ;;
    --wandb-mode)
      WANDB_MODE="$2"
      shift 2
      ;;
    --max-attempts)
      MAX_ATTEMPTS="$2"
      shift 2
      ;;
    --no-interpolation)
      RUN_INTERPOLATION=0
      shift
      ;;
    --test-only)
      TEST_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown option: $1"
      ;;
  esac
done

expanded_seeds=()
for seed in "${SEEDS[@]}"; do
  seed="${seed//,/ }"
  read -r -a parts <<< "$seed"
  expanded_seeds+=("${parts[@]}")
done
SEEDS=("${expanded_seeds[@]}")

[[ "${#SEEDS[@]}" -gt 0 ]] || die "at least one seed is required"
command -v sbatch >/dev/null 2>&1 || die "sbatch was not found; run this on the Eden login node"

if [[ "$CONFIG" = /* ]]; then
  config_path="$CONFIG"
else
  config_path="$repo_dir/$CONFIG"
fi
[[ -f "$config_path" ]] || die "config not found: $config_path"

LOG_DIR="${LOG_DIR:-$repo_dir/project3_catgen/slurm/logs}"
mkdir -p "$LOG_DIR"

for seed in "${SEEDS[@]}"; do
  cfg_name="$(basename "$config_path" .yaml)"
  job_name="dcgan_${cfg_name}_s${seed}"
  job_name="${job_name//[^[:alnum:]_-]/_}"

  sbatch_args=(
    --job-name="$job_name"
    --account="$SBATCH_ACCOUNT"
    --partition="$SBATCH_PARTITION"
    --time="$SBATCH_TIME"
    --cpus-per-task="$SBATCH_CPUS"
    --mem="$SBATCH_MEM"
    --gres="gpu:rtx6000:$SBATCH_GPUS"
    --output="$LOG_DIR/%x_%j.out"
    --error="$LOG_DIR/%x_%j.err"
    --export=ALL,REPO_DIR="$repo_dir",CONFIG="$config_path",SEED="$seed",ATTEMPT=1,MAX_ATTEMPTS="$MAX_ATTEMPTS",TRAIN_TIMEOUT="$TRAIN_TIMEOUT",WANDB_MODE="$WANDB_MODE",RUN_INTERPOLATION="$RUN_INTERPOLATION",SBATCH_ACCOUNT="$SBATCH_ACCOUNT",SBATCH_PARTITION="$SBATCH_PARTITION",SBATCH_TIME="$SBATCH_TIME",SBATCH_GPUS="$SBATCH_GPUS",SBATCH_CPUS="$SBATCH_CPUS",SBATCH_MEM="$SBATCH_MEM",LOG_DIR="$LOG_DIR",SBATCH_JOB_NAME="$job_name"
  )

  if [[ "$TEST_ONLY" == "1" ]]; then
    echo "[test] $job_name seed=$seed config=$config_path"
    sbatch --test-only "${sbatch_args[@]}" "$script_dir/dcgan_train_24h.sbatch"
  else
    echo "[submit] $job_name seed=$seed config=$config_path"
    sbatch "${sbatch_args[@]}" "$script_dir/dcgan_train_24h.sbatch"
  fi
done
