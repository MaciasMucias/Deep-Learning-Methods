# deep-learning-methods

Deep Learning Methods for University

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Setup

```bash
git clone https://github.com/MaciasMucias/Deep-Learning-Methods.git
cd Deep-Learning-Methods
uv sync --all-packages
```

## Running an experiment from the workspace root

```bash
uv run project1-cinic10/srd/project1-cinic10/experiments/train.py
```

## Extending the project

**Adding an external package** — run from inside the project folder, not the root:
```bash
cd project1_cinic10
uv add torchvision
```
This ensures it goes into that project's `pyproject.toml`, not the workspace root.

**Adding a dependency to the shared library** — run from `dl_base/`:
```bash
cd dl_base
uv add numpy
```
All projects that depend on `dl_base` will inherit it automatically.

**After any `pyproject.toml` change**, run `uv sync` from the workspace root to update the lockfile and `.venv`.

**Dev-only dependencies** (pytest, black, etc.) — use `--dev` so they don't leak into production installs:
```bash
uv add --dev pytest
```

**Importing from the shared library** in your project code just works as a normal import — no path hacks needed:

```python
from dl_base.runner import Trainer
```

**If `uv sync` fails after pulling** someone else's changes, it usually means the `uv.lock` was updated — just re-run `uv sync` and it'll reconcile.
