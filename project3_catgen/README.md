# project3-catgen

Cat image generation with DCGAN and VAE. Michał's part is implemented as the DCGAN training pipeline plus latent interpolation.

## Train DCGAN

From the workspace root:

```powershell
uv run --package project3-catgen python -m project3_catgen.experiments.train --config project3_catgen/configs/dcgan/baseline.yaml --seeds 42
```

Outputs go to `project3_catgen/runs/<run_name>_seed(<seed>)/`.

Useful options:

```powershell
uv run --package project3-catgen python -m project3_catgen.experiments.train --config project3_catgen/configs/dcgan/baseline.yaml --seeds 42 --resume
uv run --package project3-catgen python project3_catgen/configs/dcgan/run_all.py --seeds 42
```

The CLI defaults to offline W&B. Add `--wandb-mode online` if you want cloud logging.

## Latent Interpolation

Run after training has produced `last.pth`:

```powershell
uv run --package project3-catgen python -m project3_catgen.experiments.interpolate --config project3_catgen/configs/dcgan/baseline.yaml --seed 42
```

This saves `interpolation.png` and the latent vectors next to the checkpoint.
