#!/usr/bin/bash

# shellcheck disable=SC2034
export WANDB_MODE=offline

WANDB_MODE=offline uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_bs1.yaml
uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_bs2.yaml
uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_bs3.yaml

uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_do1.yaml
uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_do2.yaml
uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_do3.yaml

uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_lr1.yaml
uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_lr2.yaml
uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_lr3.yaml

uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_wd1.yaml
uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_wd2.yaml
uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_wd3.yaml

uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_aug_n.yaml

uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_aug_hf.yaml
uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_aug_rc.yaml
uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_aug_rot.yaml
uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_aug_hf_rc.yaml
uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_aug_hf_rot.yaml
uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_aug_rc_rot.yaml
uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_aug_hf_rc_rot.yaml

uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_aug_hf_co.yaml
uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_aug_rc_co.yaml
uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_aug_rot_co.yaml
uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_aug_hf_rc_co.yaml
uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_aug_hf_rot_co.yaml
uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_aug_rc_rot_co.yaml
uv run project1_cinic10/src/project1_cinic10/experiments/train.py --seeds 0 1 2 --config project1_cinic10/configs/mobilenetv2/mobilenetv2_aug_hf_rc_rot_co.yaml