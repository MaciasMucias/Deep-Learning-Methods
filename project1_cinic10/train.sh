#!/usr/bin/bash

# shellcheck disable=SC2034
WANDB_MODE=offline

cd project1_cinic10 || exit

uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_bs1.yaml
uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_bs2.yaml
uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_bs3.yaml

uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_do1.yaml
uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_do2.yaml
uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_do3.yaml

uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_lr1.yaml
uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_lr2.yaml
uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_lr3.yaml

uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_wd1.yaml
uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_wd2.yaml
uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_wd3.yaml

uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_aug_n.yaml

uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_aug_hf.yaml
uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_aug_rc.yaml
uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_aug_rot.yaml
uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_aug_hf_rc.yaml
uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_aug_hf_rot.yaml
uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_aug_rc_rot.yaml
uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_aug_hf_rc_rot.yaml

uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_aug_hf_co.yaml
uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_aug_rc_co.yaml
uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_aug_rot_co.yaml
uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_aug_hf_rc_co.yaml
uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_aug_hf_rot_co.yaml
uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_aug_rc_rot_co.yaml
uv run src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config configs/mobilenetv2/mobilenetv2_aug_hf_rc_rot_co.yaml