#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/rtdetrv2/rtdetrv2_r18vd_binary_person_coco128.yml}"
CHECKPOINT="${CHECKPOINT:-output/rtdetrv2_r18vd_binary_person_coco128/best.pth}"
INPUT="${INPUT:-dataset/coco128_binary/images/train2017/000000000086.jpg}"
OUTPUT_DIR="${OUTPUT_DIR:-output/inference_binary_person}"
DEVICE="${DEVICE:-cuda:0}"
THRESHOLD="${THRESHOLD:-0.3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

"${PYTHON_BIN}" tools/infer_binary.py \
  -c "${CONFIG}" \
  -r "${CHECKPOINT}" \
  -i "${INPUT}" \
  -o "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --threshold "${THRESHOLD}" \
  "$@"
