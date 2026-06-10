#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash project3_catgen/slurm/submit_dcgan_sweep.sh [options passed to submit_dcgan_failsafe.sh]

Useful examples:
  bash project3_catgen/slurm/submit_dcgan_sweep.sh --seeds 42
  bash project3_catgen/slurm/submit_dcgan_sweep.sh --seeds 0 1 2 --test-only
  bash project3_catgen/slurm/submit_dcgan_sweep.sh --config-dir project3_catgen/configs/dcgan --seeds 42 --mem 48G

Options handled here:
  --config-dir DIR      Directory with *.yaml configs. Default: project3_catgen/configs/dcgan
  -h, --help           Show this help

All other options are forwarded to submit_dcgan_failsafe.sh.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_dir="$(cd "$script_dir/../.." && pwd -P)"
CONFIG_DIR="$repo_dir/project3_catgen/configs/dcgan"
forward_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-dir)
      if [[ "$2" = /* ]]; then
        CONFIG_DIR="$2"
      else
        CONFIG_DIR="$repo_dir/$2"
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      forward_args+=("$1")
      shift
      ;;
  esac
done

[[ -d "$CONFIG_DIR" ]] || { echo "[error] config dir not found: $CONFIG_DIR" >&2; exit 1; }

shopt -s nullglob
configs=("$CONFIG_DIR"/*.yaml)
shopt -u nullglob

[[ "${#configs[@]}" -gt 0 ]] || { echo "[error] no YAML configs found in $CONFIG_DIR" >&2; exit 1; }

for config in "${configs[@]}"; do
  echo "[sweep] submitting $(basename "$config")"
  bash "$script_dir/submit_dcgan_failsafe.sh" --config "$config" "${forward_args[@]}"
done
