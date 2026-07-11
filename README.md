# RT-DETRv2 Binary Fine-Tuning Workflow

This repository contains a PyTorch RT-DETRv2 fine-tuning workflow for a
binary object detection task: detect one target class, `person`, or treat the
image/region as background.

The implementation is based on the upstream `rtdetrv2_pytorch` project and adds
a small, reproducible COCO128 conversion path for quick verification.

## Citation and Upstream References

This fine-tuning repository is built on the official RT-DETR implementation:

- Original code: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- Original RT-DETR paper: [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069), CVPR 2024
- RT-DETRv2 paper/report: [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](https://arxiv.org/abs/2407.17140)

## What Changed

The detector solver was not changed. Training and evaluation still use the
original RT-DETRv2 detection pipeline:

- Solver: `rtdetrv2_pytorch/src/solver/det_solver.py`
- Training loop: `rtdetrv2_pytorch/src/solver/det_engine.py`
- Evaluator setup: `rtdetrv2_pytorch/src/core/yaml_config.py`
- COCO evaluator: `rtdetrv2_pytorch/src/data/dataset/coco_eval.py`

The changes are additive and live in configs, dataset preparation, inference
convenience, and documentation.

| File | Purpose |
| --- | --- |
| `rtdetrv2_pytorch/tools/prepare_binary_coco128.py` | Downloads COCO128, reads YOLO labels, keeps only `person` boxes, remaps them to class id `0`, preserves no-person images as negative samples, and writes COCO JSON annotations. |
| `rtdetrv2_pytorch/configs/dataset/coco128_person_detection.yml` | Defines the one-class COCO dataset for RT-DETRv2 with `num_classes: 1` and `remap_mscoco_category: False`. |
| `rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_binary_person_coco128.yml` | Defines the fine-tuning smoke config based on RT-DETRv2 R18. It uses a COCO checkpoint for tuning and a CPU-friendly `160x160` verification size. |
| `rtdetrv2_pytorch/references/deploy/rtdetrv2_torch.py` | Updated to run from the repo root, use the config's `eval_spatial_size`, and accept `--threshold`. |
| `rtdetrv2_pytorch/tools/infer_binary.py` | New inference-only script that loads a trained checkpoint, runs a file or directory of images, and writes rendered predictions plus JSON outputs. |
| `rtdetrv2_pytorch/scripts/train_single_gpu.sh` | New single-GPU training bash wrapper. |
| `rtdetrv2_pytorch/scripts/infer_single_gpu.sh` | New single-GPU inference bash wrapper. |
| `docs/binary_rtdetrv2_implementation.html` | HTML summary of the workflow, changed files, and verification results. |
| `.gitignore`, `rtdetrv2_pytorch/.gitignore` | Prevents local datasets, checkpoints, virtualenvs, outputs, and OS files from being uploaded. |

## Binary Dataset Design

The custom dataset is still an object detection dataset, not an image-level
classifier. The binary behavior is implemented as:

- Foreground class: `person`
- Foreground label id: `0`
- All other COCO classes are removed from annotations
- Images without `person` annotations are preserved as negative/background
  examples

Generated files:

```text
rtdetrv2_pytorch/dataset/coco128_binary/
├── annotations/
│   ├── person_train.json
│   └── person_val.json
└── images/
    └── train2017/
```

The generated dataset is intentionally ignored by git because it is
reproducible from the preparation script.

## Environment Setup

From the repository root:

```bash
cd rtdetrv2_pytorch
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

On Apple Silicon or CPU-only environments, if the full requirements are too
heavy, the verified minimum for the smoke workflow was:

```bash
.venv/bin/python -m pip install torch torchvision PyYAML faster-coco-eval scipy tensorboard
```

## Prepare the Custom Binary Dataset

Run:

```bash
cd rtdetrv2_pytorch
.venv/bin/python tools/prepare_binary_coco128.py --force
```

Expected result:

```text
Prepared binary COCO128 dataset: 103 train images, 25 val images, 254 person boxes
```

Important checks performed by the script:

- COCO128 is downloaded and extracted
- Only one category is written: `id: 0`, `name: person`
- Train and validation splits both contain positive and negative images
- Converted bounding boxes have valid dimensions
- Annotation image references are valid

To convert a different COCO class later:

```bash
.venv/bin/python tools/prepare_binary_coco128.py --target-class car --force
```

If the target class changes, also update the dataset config annotation filenames
and category naming expectations.

## Fine-Tune for 1 Epoch

The included config is a fast CPU smoke config. It uses `160x160` images so the
pipeline can be verified quickly without a GPU.

Run:

```bash
cd rtdetrv2_pytorch
.venv/bin/python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_binary_person_coco128.yml \
  --device cpu \
  --seed 0
```

Single-GPU bash wrapper:

```bash
cd rtdetrv2_pytorch
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 bash scripts/train_single_gpu.sh
```

You can override the config, device, seed, or pass extra `tools/train.py`
arguments:

```bash
CONFIG=configs/rtdetrv2/rtdetrv2_r18vd_binary_person_coco128.yml \
DEVICE=cuda:0 \
SEED=0 \
bash scripts/train_single_gpu.sh --use-amp
```

The config already includes:

```yaml
tuning: https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
epoches: 1
num_classes: 1
eval_spatial_size: [160, 160]
```

Training outputs are written to:

```text
rtdetrv2_pytorch/output/rtdetrv2_r18vd_binary_person_coco128/
├── best.pth
├── checkpoint0000.pth
├── last.pth
├── log.txt
└── eval/
```

These outputs are ignored by git.

## Evaluation / Testing

To evaluate a trained checkpoint on the binary validation split:

```bash
cd rtdetrv2_pytorch
.venv/bin/python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_binary_person_coco128.yml \
  -r output/rtdetrv2_r18vd_binary_person_coco128/best.pth \
  --test-only \
  --device cpu
```

This uses the original `CocoEvaluator` with:

```yaml
evaluator:
  type: CocoEvaluator
  iou_types: ['bbox']
```

Evaluation is therefore standard COCO bbox evaluation, but against the
one-class validation JSON:

```text
dataset/coco128_binary/annotations/person_val.json
```

The latest verified 1-epoch CPU smoke run produced:

```text
AP@[IoU=0.50:0.95] = 0.059
AP@0.50             = 0.083
AR@maxDets=100      = 0.195
```

These numbers are low because the run is intentionally tiny: one epoch,
COCO128, CPU, and `160x160`. They confirm the pipeline works; they are not
production-quality model metrics.

## Inference Smoke Test

Run inference on one validation image with the inference-only Python script:

```bash
cd rtdetrv2_pytorch
.venv/bin/python tools/infer_binary.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_binary_person_coco128.yml \
  -r output/rtdetrv2_r18vd_binary_person_coco128/best.pth \
  -i dataset/coco128_binary/images/train2017/000000000086.jpg \
  -o output/inference_binary_person \
  --device cpu \
  --threshold 0.01
```

Single-GPU inference bash wrapper:

```bash
cd rtdetrv2_pytorch
CUDA_VISIBLE_DEVICES=0 \
DEVICE=cuda:0 \
CHECKPOINT=output/rtdetrv2_r18vd_binary_person_coco128/best.pth \
INPUT=dataset/coco128_binary/images/train2017/000000000086.jpg \
THRESHOLD=0.3 \
bash scripts/infer_single_gpu.sh
```

The inference-only script writes one JSON file and one rendered image per input
image:

```text
output/inference_binary_person/
├── 000000000086.json
└── 000000000086_pred.jpg
```

The older deploy reference script is still available:

```bash
cd rtdetrv2_pytorch
.venv/bin/python references/deploy/rtdetrv2_torch.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_binary_person_coco128.yml \
  -r output/rtdetrv2_r18vd_binary_person_coco128/best.pth \
  -f dataset/coco128_binary/images/train2017/000000000086.jpg \
  -d cpu \
  --threshold 0.01
```

The script writes:

```text
results_0.jpg
```

For the smoke checkpoint, predicted labels were verified to be only class `0`.

## Using a Larger Real Training Setup

The current config is deliberately small for fast verification. For real model
quality, use a larger dataset and adjust:

```yaml
eval_spatial_size: [640, 640]
...
- {type: Resize, size: [640, 640]}
epoches: 50
```

Then train on GPU if available:

```bash
cd rtdetrv2_pytorch
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 bash scripts/train_single_gpu.sh
```

For multi-GPU training, follow the upstream RT-DETRv2 `torchrun` style:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --master_port=9909 \
  --nproc_per_node=4 \
  tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_binary_person_coco128.yml \
  --use-amp \
  --seed 0
```

## Notes

- No solver changes were made.
- No evaluation logic changes were made.
- Binary behavior comes from the generated annotations plus `num_classes: 1`.
- Generated datasets, checkpoints, outputs, and local environments are excluded
  from git and should be regenerated locally.
