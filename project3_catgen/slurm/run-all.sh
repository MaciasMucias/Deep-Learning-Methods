#!/usr/bin/env bash

#SBATCH --job-name=catgen-dcgan
#SBATCH --array=0-23%20
#SBATCH --output=project3_catgen/slurm/logs/output_%A_%a.out
#SBATCH --error=project3_catgen/slurm/logs/error_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=24:00:00
#SBATCH --partition=student
#SBATCH --account=stud-2526-l-03
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
PARAMS_FILE="${PARAMS_FILE:-project3_catgen/slurm/params.sh}"

cd "$REPO_DIR"
mkdir -p project3_catgen/slurm/logs

PARAMS_RAW="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PARAMS_FILE")"
PARAMS_CLEAN="$(echo "$PARAMS_RAW" | tr -d '"')"
read -r -a PARAMS <<< "$PARAMS_CLEAN"

date
echo "SLURMD_NODENAME: ${SLURMD_NODENAME:-unknown}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-none}"
echo "Params file: $PARAMS_FILE"
echo "Training params: $PARAMS_CLEAN"

bash project3_catgen/slurm/run.sh "${PARAMS[@]}"
date
