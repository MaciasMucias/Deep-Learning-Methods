#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-eden}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-~/Deep-Learning-Methods}"
LOCAL_RESULTS_DIR="${LOCAL_RESULTS_DIR:-project3_catgen/results_from_eden}"

mkdir -p "$LOCAL_RESULTS_DIR/runs" "$LOCAL_RESULTS_DIR/logs"

rsync -vazP \
  --exclude='wandb/' \
  --exclude='*.tmp' \
  "$REMOTE:$REMOTE_PROJECT_DIR/project3_catgen/runs/" \
  "$LOCAL_RESULTS_DIR/runs/"

rsync -vazP \
  "$REMOTE:$REMOTE_PROJECT_DIR/project3_catgen/slurm/logs/" \
  "$LOCAL_RESULTS_DIR/logs/"
