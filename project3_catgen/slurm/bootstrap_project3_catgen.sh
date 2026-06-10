#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/MaciasMucias/Deep-Learning-Methods.git}"
BRANCH="${BRANCH:-}"
WORK_ROOT="${WORK_ROOT:-$HOME/dl_project3}"
DATASET_SLUG="${DATASET_SLUG:-crawford/cat-dataset}"
UV_PYTHON="${UV_PYTHON:-3.13}"
UV_SYNC_ARGS="${UV_SYNC_ARGS:---frozen}"

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

ensure_kaggle() {
  if command -v kaggle >/dev/null 2>&1; then
    return
  fi
  find_python
  echo "[setup] kaggle CLI not found; installing it into the user site-packages"
  "$PYTHON_BIN" -m pip install --user kaggle
  export PATH="$HOME/.local/bin:$PATH"
  command -v kaggle >/dev/null 2>&1 || die "kaggle was installed but is still not on PATH"
}

has_images() {
  find "$1" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -print -quit 2>/dev/null | grep -q .
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
candidate_repo="$(cd "$script_dir/../.." 2>/dev/null && pwd -P || true)"

if [[ -f "$candidate_repo/pyproject.toml" && -d "$candidate_repo/project3_catgen" ]]; then
  default_repo_dir="$candidate_repo"
else
  default_repo_dir="$WORK_ROOT/Deep-Learning-Methods"
fi

REPO_DIR="${REPO_DIR:-$default_repo_dir}"
DATA_DIR="${DATA_DIR:-$REPO_DIR/project3_catgen/data/cats}"

mkdir -p "$WORK_ROOT"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  command -v git >/dev/null 2>&1 || die "git is required"
  echo "[repo] cloning $REPO_URL into $REPO_DIR"
  if [[ -n "$BRANCH" ]]; then
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
  else
    git clone "$REPO_URL" "$REPO_DIR"
  fi
else
  echo "[repo] using existing repository: $REPO_DIR"
  git -C "$REPO_DIR" pull --ff-only
fi

cd "$REPO_DIR"

ensure_uv

echo "[python] preparing uv environment for project3-catgen"
if [[ -n "$UV_PYTHON" ]]; then
  uv python install "$UV_PYTHON" || echo "[warn] uv could not install Python $UV_PYTHON; trying to use an existing interpreter"
  uv sync --package project3-catgen --python "$UV_PYTHON" $UV_SYNC_ARGS
else
  uv sync --package project3-catgen $UV_SYNC_ARGS
fi

if [[ ! -f "$HOME/.kaggle/kaggle.json" && ( -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ) ]]; then
  cat >&2 <<'EOF'
[error] Kaggle credentials are missing.

Create ~/.kaggle/kaggle.json with:
  {"username":"YOUR_USERNAME","key":"YOUR_KEY"}

Then run:
  chmod 600 ~/.kaggle/kaggle.json

Alternatively export KAGGLE_USERNAME and KAGGLE_KEY before running this script.
EOF
  exit 1
fi

ensure_kaggle

mkdir -p "$DATA_DIR"
if [[ "${FORCE_DATA_DOWNLOAD:-0}" != "1" ]] && has_images "$DATA_DIR"; then
  echo "[data] images already found in $DATA_DIR; skipping download"
else
  echo "[data] downloading Kaggle dataset $DATASET_SLUG into $DATA_DIR"
  kaggle datasets download -d "$DATASET_SLUG" -p "$DATA_DIR" --unzip
fi

image_count="$(find "$DATA_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l | tr -d '[:space:]')"
echo "[data] found $image_count image files under $DATA_DIR"

cat <<EOF

[ok] Project 3 is prepared.

Repository:
  $REPO_DIR

Dataset:
  $DATA_DIR

Submit one DCGAN run:
  cd "$REPO_DIR"
  bash project3_catgen/slurm/submit_dcgan_failsafe.sh --config project3_catgen/configs/dcgan/baseline.yaml --seed 42

Submit all DCGAN hyperparameter configs:
  cd "$REPO_DIR"
  bash project3_catgen/slurm/submit_dcgan_sweep.sh --seeds 42
EOF
