# RT-DETRv2 Binary Fine-Tuning Workflow

This repository contains a PyTorch RT-DETRv2 fine-tuning workflow for a
binary object detection task: detect one target class, `person`, or treat the
image/region as background.

The implementation is based on the upstream `rtdetrv2_pytorch` project and adds
a small, reproducible custom dataset conversion path for quick verification.

## Citation and Upstream References

This fine-tuning repository is built on the official RT-DETR implementation:

- Original code: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- Original RT-DETR paper: [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069), CVPR 2024
- RT-DETRv2 paper/report: [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](https://arxiv.org/abs/2407.17140)

If you use RT-DETR or RT-DETRv2 in your work, please use the following BibTeX entries:

```bibtex
@misc{lv2023detrs,
      title={DETRs Beat YOLOs on Real-time Object Detection},
      author={Yian Zhao and Wenyu Lv and Shangliang Xu and Jinman Wei and Guanzhong Wang and Qingqing Dang and Yi Liu and Jie Chen},
      year={2023},
      eprint={2304.08069},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{lv2024rtdetrv2improvedbaselinebagoffreebies,
      title={RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer},
      author={Wenyu Lv and Yian Zhao and Qinyao Chang and Kui Huang and Guanzhong Wang and Yi Liu},
      year={2024},
      eprint={2407.17140},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.17140},
}
```

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
| `rtdetrv2_pytorch/tools/prepare_binary_custom.py` | Converts a custom YOLO-format dataset, keeps one target class, remaps it to class id `0`, preserves no-target images as negative samples, and writes COCO JSON annotations. |
| `rtdetrv2_pytorch/configs/dataset/custom_person_detection.yml` | Defines the one-class COCO dataset for RT-DETRv2 with `num_classes: 1` and `remap_mscoco_category: False`. |
| `rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_binary_person_custom.yml` | Defines the fine-tuning smoke config based on RT-DETRv2 R18. It uses a COCO checkpoint for tuning and a CPU-friendly `160x160` verification size. |
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
- All non-target classes are removed from annotations
- Images without `person` annotations are preserved as negative/background
  examples

Generated files:

```text
rtdetrv2_pytorch/dataset/custom_binary/
├── annotations/
│   ├── person_train.json
│   └── person_val.json
└── images/
    ├── train/
    └── val/
```

The generated dataset is intentionally ignored by git because it is
reproducible from the preparation script.


## Custom Dataset Folder and Annotation Format

Prepare your own dataset in YOLO detection format first. The recommended input
folder is:

```text
rtdetrv2_pytorch/dataset/custom_raw/
├── images/
│   ├── train/
│   │   ├── person_001.jpg
│   │   └── no_person_001.jpg
│   └── val/
│       ├── person_101.jpg
│       └── no_person_101.jpg
└── labels/
    ├── train/
    │   ├── person_001.txt
    │   └── no_person_001.txt
    └── val/
        ├── person_101.txt
        └── no_person_101.txt
```

The script also accepts a single unsplit folder and will create deterministic
train/val splits:

```text
rtdetrv2_pytorch/dataset/custom_raw/
├── images/
│   ├── person_001.jpg
│   ├── no_person_001.jpg
│   └── person_002.jpg
└── labels/
    ├── person_001.txt
    ├── no_person_001.txt
    └── person_002.txt
```

Each label file must have the same basename as the image. For example,
`images/train/person_001.jpg` uses `labels/train/person_001.txt`.
Nested image folders are allowed as long as the same relative path exists under
`labels/`.

Each annotation line must use YOLO normalized `xywh` format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Values after `class_id` must be normalized to the range `0.0` to `1.0`, relative
to image width and height. For this binary fine-tuning workflow, choose the
single target class id you want to detect and pass it with `--target-class-id`.
That class is remapped to COCO category id `0` in the converted output.

Positive image example, `labels/train/person_001.txt`:

```text
0 0.5125 0.4380 0.2100 0.4960
0 0.3020 0.5170 0.1200 0.3380
```

Negative/background image example, `labels/train/no_person_001.txt`:

```text

```

A missing label file is also treated as a negative/background image. Keeping
negative images is recommended because the script verifies that train and val
splits contain both positive and background examples.

After conversion, RT-DETRv2 trains from COCO-style JSON files:

```text
rtdetrv2_pytorch/dataset/custom_binary/
├── annotations/
│   ├── person_train.json
│   └── person_val.json
└── images/
    ├── train/
    │   ├── person_001.jpg
    │   └── no_person_001.jpg
    └── val/
        ├── person_101.jpg
        └── no_person_101.jpg
```

Example converted COCO annotation:

```json
{
  "images": [
    {"id": 1, "file_name": "train/person_001.jpg", "width": 1280, "height": 720}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [521.6, 136.8, 268.8, 357.12],
      "area": 95954.176,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 0, "name": "person", "supercategory": "object"}
  ]
}
```

Important difference: the source annotations are YOLO text files, but the
training config reads the generated COCO JSON files. You do not manually write
COCO JSON unless you want to bypass `tools/prepare_binary_custom.py`.

## Environment Setup

This workflow assumes you use your local Python environment directly. From the repository root:

```bash
cd rtdetrv2_pytorch
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

On Apple Silicon or CPU-only environments, if the full requirements are too
heavy, the verified minimum for the smoke workflow was:

```bash
python3 -m pip install torch torchvision Pillow PyYAML faster-coco-eval scipy tensorboard
```

## Prepare the Custom Binary Dataset

Convert your own YOLO-format dataset into the COCO JSON layout used by the
RT-DETRv2 config:

```bash
cd rtdetrv2_pytorch
python3 tools/prepare_binary_custom.py \
  --source-root dataset/custom_raw \
  --target-class-id 0 \
  --target-class-name person \
  --force
```

Expected result:

```text
Prepared custom binary dataset: <train_count> train images, <val_count> val images, <box_count> person boxes
Source root: .../rtdetrv2_pytorch/dataset/custom_raw
Output root: .../rtdetrv2_pytorch/dataset/custom_binary
```

Important checks performed by the script:

- `images/` and `labels/` exist under your custom source root
- Only the requested YOLO class id is kept
- The kept class is written as one category: `id: 0`, `name: person`
- Train and validation splits contain positive and negative images
- Converted bounding boxes have valid dimensions
- Annotation image references are valid

To convert a different target class later:

```bash
python3 tools/prepare_binary_custom.py \
  --source-root dataset/custom_raw \
  --target-class-id 2 \
  --target-class-name car \
  --force
```

If `--target-class-name` changes, also update the annotation filenames in
`configs/dataset/custom_person_detection.yml`, for example from
`person_train.json` to `car_train.json`.

For a quick smoke dataset only, you can run:

```bash
python3 tools/prepare_binary_custom.py --download-sample --force
```

## Fine-Tune for 1 Epoch

The included config is a fast CPU smoke config. It uses `160x160` images so the
pipeline can be verified quickly without a GPU.

Run:

```bash
cd rtdetrv2_pytorch
python3 tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_binary_person_custom.yml \
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
CONFIG=configs/rtdetrv2/rtdetrv2_r18vd_binary_person_custom.yml \
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
rtdetrv2_pytorch/output/rtdetrv2_r18vd_binary_person_custom/
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
python3 tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_binary_person_custom.yml \
  -r output/rtdetrv2_r18vd_binary_person_custom/best.pth \
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
dataset/custom_binary/annotations/person_val.json
```

The latest verified 1-epoch CPU smoke run used a four-image synthetic custom
YOLO dataset and produced:

```text
AP@[IoU=0.50:0.95] = 0.003
AP@0.50             = 0.015
AR@maxDets=100      = 0.200
```

These numbers are low because the run is intentionally tiny: one epoch, four
custom images, CPU, and `160x160`. They confirm the fine-tuning and evaluation
pipeline works; they are not production-quality model metrics.

## Inference Smoke Test

Run inference on one validation image with the inference-only Python script:

```bash
cd rtdetrv2_pytorch
python3 tools/infer_binary.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_binary_person_custom.yml \
  -r output/rtdetrv2_r18vd_binary_person_custom/best.pth \
  -i dataset/custom_binary/images/val/person_101.jpg \
  -o output/inference_binary_person \
  --device cpu \
  --threshold 0.01
```

Single-GPU inference bash wrapper:

```bash
cd rtdetrv2_pytorch
CUDA_VISIBLE_DEVICES=0 \
DEVICE=cuda:0 \
CHECKPOINT=output/rtdetrv2_r18vd_binary_person_custom/best.pth \
INPUT=dataset/custom_binary/images/val/person_101.jpg \
THRESHOLD=0.3 \
bash scripts/infer_single_gpu.sh
```

The inference-only script writes one JSON file and one rendered image per input
image:

```text
output/inference_binary_person/
├── person_101.json
└── person_101_pred.jpg
```

The older deploy reference script is still available:

```bash
cd rtdetrv2_pytorch
python3 references/deploy/rtdetrv2_torch.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_binary_person_custom.yml \
  -r output/rtdetrv2_r18vd_binary_person_custom/best.pth \
  -f dataset/custom_binary/images/val/person_101.jpg \
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
  -c configs/rtdetrv2/rtdetrv2_r18vd_binary_person_custom.yml \
  --use-amp \
  --seed 0
```

## Notes

- No solver changes were made.
- No evaluation logic changes were made.
- Binary behavior comes from the generated annotations plus `num_classes: 1`.
- Generated datasets, checkpoints, outputs, and local environments are excluded
  from git and should be regenerated locally.
