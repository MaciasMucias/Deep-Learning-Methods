# Project 3 Slurm workflow

This folder contains Eden/Slurm scripts for Michał Bober's Project 3 part:
DCGAN training, checkpoint resume, hyperparameter runs, and latent interpolation.

## Requirement check

From the course brief, Project 3 requires cat image generation, hyperparameter
experiments, quantitative comparison such as FID, qualitative assessment, mode
collapse discussion, and latent interpolation. The plan assigns these tasks:

- Michał Bober: DCGAN implementation.
- Michał Bober: latent interpolation experiment.
- Both: hyperparameter experiments, results analysis, report, slides.
- Maciej Karaśkiewicz: preprocessing, VAE, FID pipeline.

Current `project3_catgen` status for Michał's part:

- DCGAN generator/discriminator are implemented in `src/project3_catgen/models/dcgan.py`.
- Training loop, W&B logging, sample images, checkpoint save/load, and RNG resume are implemented in `src/project3_catgen/dcgan_trainer.py`.
- The training CLI supports `--resume` in `src/project3_catgen/experiments/train.py`.
- Hyperparameter YAMLs exist in `configs/dcgan/`.
- Latent interpolation CLI exists in `src/project3_catgen/experiments/interpolate.py`.
- FID helper code exists in `src/project3_catgen/fid.py`, but there is no complete Slurm/report workflow for FID in this folder.
- VAE and dogs-vs-cats extension are not part of this Slurm workflow.

So: the owned DCGAN + interpolation part is covered operationally here. Full
project scoring still needs generated results, qualitative discussion, FID/VAE
comparison from the other project part, and report/presentation text.

## Eden facts used by these scripts

- Slurm account: `stud-2526-l-03`
- Partition: `student`
- Max job time: `24:00:00`
- GPU resource: `gpu:rtx6000`

The failsafe logic uses a shorter in-job timeout, default `23h30m`, so the
script can inspect `last.pth` and submit the next job before Slurm kills it.
If a job stops mid-epoch, resume starts from the previous completed epoch,
because `last.pth` is written only after an epoch finishes.

## One-time setup

Log in to Eden, then prepare Kaggle credentials:

```bash
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

The file should contain:

```json
{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_API_KEY"}
```

Then run the bootstrap script. If you already cloned the repository:

```bash
cd ~/Deep-Learning-Methods
bash project3_catgen/slurm/bootstrap_project3_catgen.sh
```

If you downloaded only this script folder, the bootstrap script clones:

```bash
WORK_ROOT=$HOME/dl_project3 bash bootstrap_project3_catgen.sh
```

The bootstrap script:

- clones or updates `https://github.com/MaciasMucias/Deep-Learning-Methods.git`,
- installs `uv` if needed,
- prepares the `project3-catgen` environment,
- downloads `crawford/cat-dataset` with the Kaggle CLI,
- unzips it under `project3_catgen/data/cats`.

## Manual dataset upload

Use this route if downloading from Kaggle directly on Eden is inconvenient or
Kaggle credentials are not configured on the cluster.

First prepare the repository and Python environment on Eden:

```bash
git clone https://github.com/MaciasMucias/Deep-Learning-Methods.git
cd Deep-Learning-Methods

# If uv is missing:
python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"

# The project currently requires Python 3.13.
uv python install 3.13
uv sync --package project3-catgen --python 3.13 --frozen
```

Download the dataset ZIP on your own device from:

```text
https://www.kaggle.com/datasets/crawford/cat-dataset
```

Assume the downloaded file is named `cat-dataset.zip`.

If you are inside the MINI PW network, upload it directly to Eden:

```bash
scp cat-dataset.zip <login>@eden.mini.pw.edu.pl:~/cat-dataset.zip
```

If you are outside the MINI PW network, use the SSH jump host:

```bash
scp -J <login>@ssh.mini.pw.edu.pl cat-dataset.zip <login>@eden.mini.pw.edu.pl:~/cat-dataset.zip
```

On Windows PowerShell, the local path may look like this:

```powershell
scp .\cat-dataset.zip <login>@eden.mini.pw.edu.pl:~/cat-dataset.zip
scp -J <login>@ssh.mini.pw.edu.pl .\cat-dataset.zip <login>@eden.mini.pw.edu.pl:~/cat-dataset.zip
```

Then log in to Eden and unpack it into the directory used by the YAML configs:

```bash
ssh <login>@eden.mini.pw.edu.pl
cd ~/Deep-Learning-Methods

mkdir -p project3_catgen/data/cats
unzip -q ~/cat-dataset.zip -d project3_catgen/data/cats
```

The training loader searches recursively for `.jpg`, `.jpeg`, and `.png` files,
so the original directory layout from the ZIP can stay as-is. Verify that images
are visible:

```bash
find project3_catgen/data/cats -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l
```

If the command prints `0`, the ZIP probably unpacked one level too high or too
low. Move the directory containing the image files under:

```text
project3_catgen/data/cats/
```

After this manual step, submit training normally; the Slurm scripts do not need
Kaggle credentials once the images are already in `project3_catgen/data/cats`.

## Submit one training run

From the repository root:

```bash
bash project3_catgen/slurm/submit_dcgan_failsafe.sh \
  --config project3_catgen/configs/dcgan/baseline.yaml \
  --seed 42
```

Before submitting for real, ask Slurm for an estimate:

```bash
bash project3_catgen/slurm/submit_dcgan_failsafe.sh \
  --config project3_catgen/configs/dcgan/baseline.yaml \
  --seed 42 \
  --test-only
```

Useful overrides:

```bash
bash project3_catgen/slurm/submit_dcgan_failsafe.sh \
  --config project3_catgen/configs/dcgan/latent_dim_256.yaml \
  --seeds 0 1 2 \
  --mem 48G \
  --cpus 8 \
  --train-timeout 23h30m \
  --wandb-mode offline
```

Each seed is submitted as an independent Slurm job. Outputs go to:

```text
project3_catgen/runs/<run_name>_seed(<seed>)/
project3_catgen/slurm/logs/
```

## Submit all DCGAN configs

```bash
bash project3_catgen/slurm/submit_dcgan_sweep.sh --seeds 42
```

For repeated experiments:

```bash
bash project3_catgen/slurm/submit_dcgan_sweep.sh --seeds 0 1 2
```

This submits one job per config per seed. Remember the cluster limit: max 20
queued/running jobs per person.

## Monitor and manage jobs

```bash
squeue -u "$USER"
sinfo
sfree
scontrol show job <job_id>
scancel <job_id>
```

Logs:

```bash
tail -f project3_catgen/slurm/logs/<job_name>_<job_id>.out
tail -f project3_catgen/slurm/logs/<job_name>_<job_id>.err
```

## What happens after 24 hours

The job script runs training through `timeout 23h30m`. After timeout:

1. it checks `project3_catgen/runs/<run_name>_seed(<seed>)/last.pth`,
2. if the checkpoint has reached `training.num_epochs`, it runs interpolation,
3. if not, it submits the same job again with `--resume`,
4. this repeats up to `--max-attempts` attempts.

The resume mechanism is epoch-level, not batch-level.
