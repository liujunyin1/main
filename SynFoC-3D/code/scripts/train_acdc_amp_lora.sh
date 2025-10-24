#!/usr/bin/env bash

# SynFoC-3D ACDC training with mixed precision and SAM LoRA finetuning on two GPUs.
# Adjust DATA_ROOT if your dataset indices point somewhere else.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
DATA_ROOT="${PROJECT_ROOT}/../data/ACDC"
WEIGHT_ROOT="${PROJECT_ROOT}/../weights"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} \
python "${PROJECT_ROOT}/train_3d.py" \
  --labeled_list "${DATA_ROOT}/labeled.txt" \
  --unlabeled_list "${DATA_ROOT}/unlabeled.txt" \
  --val_list "${DATA_ROOT}/val.txt" \
  --num_classes 4 \
  --in_channels 1 \
  --patch_size 96 128 128 \
  --spacing 1.0 1.0 1.0 \
  --train_bs 2 \
  --val_bs 2 \
  --epochs 200 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --sam_model vit_b \
  --sam_checkpoint "${WEIGHT_ROOT}/medsam_vit_b.pth" \
  --sam_pixel_mean 0.0 0.0 0.0 \
  --sam_pixel_std 1.0 1.0 1.0 \
  --sam_use_lora \
  --sam_lora_rank 4 \
  --amp \
  --outdir "${PROJECT_ROOT}/runs/syncf3d" \
  --exp_name acdc_amp_lora_dp \
  --gpu 0,1
