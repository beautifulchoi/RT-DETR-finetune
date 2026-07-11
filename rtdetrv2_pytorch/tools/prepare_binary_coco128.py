"""Prepare a one-class COCO128 detection dataset.

The output keeps only one foreground class and leaves images without that class
as negative/background examples. By default this creates a person-vs-background
dataset for quick RT-DETRv2 smoke tests.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"

COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


@dataclass(frozen=True)
class ImageRecord:
    id: int
    file_name: str
    width: int
    height: int
    source_path: Path
    annotations: list[dict]

    @property
    def is_positive(self) -> bool:
        return bool(self.annotations)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="dataset/coco128_binary", type=Path)
    parser.add_argument("--download-url", default=COCO128_URL)
    parser.add_argument("--target-class", default="person", choices=COCO80_NAMES)
    parser.add_argument("--val-fraction", default=0.2, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--force", action="store_true", help="overwrite an existing output dataset")
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    print(f"Downloading {url} -> {destination}")
    urllib.request.urlretrieve(url, destination)


def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    marker = extract_dir / "coco128"
    if marker.exists():
        return marker
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    if not marker.exists():
        raise FileNotFoundError(f"Expected extracted dataset at {marker}")
    return marker


def yolo_to_coco_bbox(values: list[float], width: int, height: int) -> list[float] | None:
    x_center, y_center, box_width, box_height = values
    x = (x_center - box_width / 2.0) * width
    y = (y_center - box_height / 2.0) * height
    w = box_width * width
    h = box_height * height

    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + w))
    y2 = max(0.0, min(float(height), y + h))
    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 0.0 or clipped_h <= 0.0:
        return None
    return [x1, y1, clipped_w, clipped_h]


def read_records(source_root: Path, target_class_id: int) -> list[ImageRecord]:
    image_dir = source_root / "images" / "train2017"
    label_dir = source_root / "labels" / "train2017"
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    records: list[ImageRecord] = []
    next_annotation_id = 1
    for image_index, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image:
            width, height = image.size

        annotations = []
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}: {line}")
                class_id = int(float(parts[0]))
                if class_id != target_class_id:
                    continue

                bbox = yolo_to_coco_bbox([float(value) for value in parts[1:]], width, height)
                if bbox is None:
                    continue
                annotations.append(
                    {
                        "id": next_annotation_id,
                        "image_id": image_index,
                        "category_id": 0,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    }
                )
                next_annotation_id += 1

        records.append(
            ImageRecord(
                id=image_index,
                file_name=image_path.name,
                width=width,
                height=height,
                source_path=image_path,
                annotations=annotations,
            )
        )
    return records


def split_records(records: list[ImageRecord], val_fraction: float, seed: int) -> tuple[list[ImageRecord], list[ImageRecord]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")

    rng = random.Random(seed)
    positives = [record for record in records if record.is_positive]
    negatives = [record for record in records if not record.is_positive]
    if not positives or not negatives:
        raise ValueError("Expected both positive and negative images after binary conversion")

    def split_group(group: list[ImageRecord]) -> tuple[list[ImageRecord], list[ImageRecord]]:
        shuffled = list(group)
        rng.shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * val_fraction))
        val_count = min(len(shuffled) - 1, val_count) if len(shuffled) > 1 else 1
        return shuffled[val_count:], shuffled[:val_count]

    train_pos, val_pos = split_group(positives)
    train_neg, val_neg = split_group(negatives)
    train = sorted(train_pos + train_neg, key=lambda record: record.id)
    val = sorted(val_pos + val_neg, key=lambda record: record.id)
    return train, val


def write_coco_json(records: list[ImageRecord], output_path: Path, class_name: str) -> None:
    images = [
        {
            "id": record.id,
            "file_name": record.file_name,
            "width": record.width,
            "height": record.height,
        }
        for record in records
    ]
    annotations = [annotation for record in records for annotation in record.annotations]
    payload = {
        "info": {"description": f"COCO128 binary {class_name} detection subset"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": class_name, "supercategory": "object"}],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def copy_images(records: list[ImageRecord], destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        destination = destination_dir / record.file_name
        if not destination.exists():
            shutil.copy2(record.source_path, destination)


def assert_dataset(records: list[ImageRecord], split_name: str) -> None:
    positives = sum(1 for record in records if record.is_positive)
    negatives = len(records) - positives
    if positives == 0 or negatives == 0:
        raise AssertionError(f"{split_name} split needs positive and negative images")
    image_ids = {record.id for record in records}
    for record in records:
        if record.width <= 0 or record.height <= 0:
            raise AssertionError(f"Invalid image size for {record.file_name}")
        for annotation in record.annotations:
            if annotation["category_id"] != 0:
                raise AssertionError(f"Unexpected category id: {annotation}")
            if annotation["image_id"] not in image_ids:
                raise AssertionError(f"Annotation references a missing image: {annotation}")
            _, _, width, height = annotation["bbox"]
            if width <= 0 or height <= 0 or annotation["area"] <= 0:
                raise AssertionError(f"Invalid bbox: {annotation}")


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    if output_root.exists() and args.force:
        shutil.rmtree(output_root)

    target_class_id = COCO80_NAMES.index(args.target_class)
    zip_path = output_root / "downloads" / "coco128.zip"
    source_parent = output_root / "source"

    download_file(args.download_url, zip_path)
    source_root = extract_zip(zip_path, source_parent)
    records = read_records(source_root, target_class_id)
    train_records, val_records = split_records(records, args.val_fraction, args.seed)

    annotation_prefix = args.target_class.replace(" ", "_")
    image_output_dir = output_root / "images" / "train2017"
    copy_images(records, image_output_dir)
    write_coco_json(train_records, output_root / "annotations" / f"{annotation_prefix}_train.json", args.target_class)
    write_coco_json(val_records, output_root / "annotations" / f"{annotation_prefix}_val.json", args.target_class)

    assert_dataset(train_records, "train")
    assert_dataset(val_records, "val")
    print(
        "Prepared binary COCO128 dataset: "
        f"{len(train_records)} train images, {len(val_records)} val images, "
        f"{sum(len(record.annotations) for record in records)} {args.target_class} boxes"
    )
    print(f"Output root: {output_root.resolve()}")


if __name__ == "__main__":
    main()
