#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CONFIG="${CONFIG:-configs/rtdetrv2/rtdetrv2_r18vd_binary_person_coco128.yml}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

"${PYTHON_BIN}" tools/train.py \
  -c "${CONFIG}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  "$@"
