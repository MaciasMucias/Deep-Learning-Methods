# Eden scripts for Project 3

Minimal flow:

```bash
ssh <login>@eden.mini.pw.edu.pl
cd ~/Deep-Learning-Methods
git pull
```

Put cat images under:

```text
project3_catgen/data/cats/
```

The loader searches this directory recursively for `.jpg`, `.jpeg`, and `.png`.

## Manual dataset upload

Download `cat-dataset.zip` from Kaggle on your own machine:

```text
https://www.kaggle.com/datasets/crawford/cat-dataset
```

Upload it to Eden:

```bash
scp cat-dataset.zip <login>@eden.mini.pw.edu.pl:~/cat-dataset.zip
```

If you are outside the MINI network, use the jump host:

```bash
scp -J <login>@ssh.mini.pw.edu.pl cat-dataset.zip <login>@eden.mini.pw.edu.pl:~/cat-dataset.zip
```

Unpack on Eden:

```bash
cd ~/Deep-Learning-Methods
mkdir -p project3_catgen/data/cats
unzip -q ~/cat-dataset.zip -d project3_catgen/data/cats
find project3_catgen/data/cats -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l
```

## One run

```bash
sbatch project3_catgen/slurm/run.sh \
  --config project3_catgen/configs/dcgan/baseline.yaml \
  --seed 42
```

## Three seeds

```bash
for seed in 42 1 3; do
  sbatch project3_catgen/slurm/run.sh \
    --config project3_catgen/configs/dcgan/baseline.yaml \
    --seed "$seed"
done
```

## All configs from params.sh

`params.sh` contains 8 DCGAN configs x 3 seeds (`42`, `1`, `3`) = 24 runs.

```bash
sbatch project3_catgen/slurm/run-all.sh
```

Submit only the first three lines, which are baseline for seeds `42`, `1`, `3`:

```bash
sbatch --array=0-2%3 project3_catgen/slurm/run-all.sh
```

Check what will run:

```bash
nl -ba project3_catgen/slurm/params.sh
```

## Monitoring

```bash
squeue -u "$USER"
tail -f project3_catgen/slurm/logs/output_<job_id>.out
tail -f project3_catgen/slurm/logs/error_<job_id>.err
```

## Resume/failsafe

`run.sh` stops training after `23h30m`, before the 24h Slurm limit. If the
configured number of epochs is not finished, it submits the same job again.
Resume uses `last.pth`, so it continues after the last completed epoch.

## Copy results back

If you have an SSH alias named `eden`:

```bash
bash project3_catgen/slurm/rsync-results.sh
```
