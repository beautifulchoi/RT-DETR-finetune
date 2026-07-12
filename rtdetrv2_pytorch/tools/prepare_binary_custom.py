"""Prepare a custom one-class detection dataset for RT-DETRv2.

The input dataset must use YOLO detection labels. The output keeps one
foreground class, remaps it to category id 0, preserves images without that
class as negative/background examples, and writes COCO-style annotations for
RT-DETRv2 fine-tuning.
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


DEFAULT_SAMPLE_DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    id: int
    split: str
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
    parser.add_argument("--source-root", type=Path, help="custom YOLO dataset root")
    parser.add_argument("--output-root", default="dataset/custom_binary", type=Path)
    parser.add_argument("--target-class-id", default=0, type=int, help="YOLO class id to keep")
    parser.add_argument("--target-class-name", default="person", help="category name written to COCO JSON")
    parser.add_argument("--val-fraction", default=0.2, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--force", action="store_true", help="overwrite an existing output dataset")
    parser.add_argument(
        "--download-sample",
        action="store_true",
        help="download a tiny public YOLO dataset only for smoke verification",
    )
    parser.add_argument("--sample-url", default=DEFAULT_SAMPLE_DATASET_URL)
    parser.add_argument(
        "--allow-missing-negatives",
        action="store_true",
        help="allow splits without negative/background images",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    print(f"Downloading {url} -> {destination}")
    urllib.request.urlretrieve(url, destination)


def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    existing_roots = sorted(path for path in extract_dir.iterdir() if path.is_dir())
    if existing_roots:
        return existing_roots[0]

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)

    extracted_roots = sorted(path for path in extract_dir.iterdir() if path.is_dir())
    if not extracted_roots:
        raise FileNotFoundError(f"No extracted dataset directory found in {extract_dir}")
    return extracted_roots[0]


def resolve_source_root(args: argparse.Namespace, output_root: Path) -> Path:
    if args.source_root:
        return args.source_root
    if not args.download_sample:
        raise ValueError("Pass --source-root for your custom dataset, or --download-sample for a smoke dataset")

    zip_path = output_root / "downloads" / "custom_source.zip"
    source_parent = output_root / "source"
    download_file(args.sample_url, zip_path)
    return extract_zip(zip_path, source_parent)


def find_split_dirs(source_root: Path) -> tuple[list[tuple[str, Path, Path]], bool]:
    images_root = source_root / "images"
    labels_root = source_root / "labels"
    if not images_root.exists() or not labels_root.exists():
        raise FileNotFoundError(f"Expected images/ and labels/ under {source_root}")

    explicit_split_sets = [("train", "train"), ("val", "val"), ("train", "train2017"), ("val", "val2017")]
    splits: list[tuple[str, Path, Path]] = []
    seen: set[str] = set()
    for output_split, source_split in explicit_split_sets:
        image_dir = images_root / source_split
        label_dir = labels_root / source_split
        if image_dir.exists() and label_dir.exists() and output_split not in seen:
            splits.append((output_split, image_dir, label_dir))
            seen.add(output_split)

    if len({split for split, _, _ in splits}) >= 2:
        return splits, False

    if splits:
        return [("all", splits[0][1], splits[0][2])], True

    image_files = [path for path in images_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    if image_files:
        return [("all", images_root, labels_root)], True

    raise FileNotFoundError(
        "Expected either images/train + labels/train, images/val + labels/val, "
        "images/train2017 + labels/train2017, or flat images/ + labels/"
    )


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


def read_records(
    image_dir: Path,
    label_dir: Path,
    output_split: str,
    target_class_id: int,
    start_image_id: int,
    start_annotation_id: int,
) -> tuple[list[ImageRecord], int, int]:
    image_paths = sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    records: list[ImageRecord] = []
    next_image_id = start_image_id
    next_annotation_id = start_annotation_id
    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size

        relative_image_path = image_path.relative_to(image_dir)
        label_path = (label_dir / relative_image_path).with_suffix(".txt")
        annotations = []
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
                        "image_id": next_image_id,
                        "category_id": 0,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    }
                )
                next_annotation_id += 1

        records.append(
            ImageRecord(
                id=next_image_id,
                split=output_split,
                file_name=str(Path(output_split) / relative_image_path),
                width=width,
                height=height,
                source_path=image_path,
                annotations=annotations,
            )
        )
        next_image_id += 1
    return records, next_image_id, next_annotation_id


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
    train = sorted([rewrite_split(record, "train") for record in train_pos + train_neg], key=lambda record: record.id)
    val = sorted([rewrite_split(record, "val") for record in val_pos + val_neg], key=lambda record: record.id)
    return train, val


def rewrite_split(record: ImageRecord, split: str) -> ImageRecord:
    current_path = Path(record.file_name)
    relative_path = Path(*current_path.parts[1:]) if len(current_path.parts) > 1 else current_path
    file_name = str(Path(split) / relative_path)
    return ImageRecord(
        id=record.id,
        split=split,
        file_name=file_name,
        width=record.width,
        height=record.height,
        source_path=record.source_path,
        annotations=record.annotations,
    )


def write_coco_json(records: list[ImageRecord], output_path: Path, class_name: str) -> None:
    id_map = {record.id: index for index, record in enumerate(records, start=1)}
    images = [
        {
            "id": id_map[record.id],
            "file_name": record.file_name,
            "width": record.width,
            "height": record.height,
        }
        for record in records
    ]

    annotations = []
    next_annotation_id = 1
    for record in records:
        for annotation in record.annotations:
            rewritten = dict(annotation)
            rewritten["id"] = next_annotation_id
            rewritten["image_id"] = id_map[record.id]
            annotations.append(rewritten)
            next_annotation_id += 1

    payload = {
        "info": {"description": f"custom binary {class_name} detection dataset"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": class_name, "supercategory": "object"}],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def copy_images(records: list[ImageRecord], destination_root: Path) -> None:
    for record in records:
        destination = destination_root / record.file_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(record.source_path, destination)


def assert_dataset(records: list[ImageRecord], split_name: str, allow_missing_negatives: bool) -> None:
    positives = sum(1 for record in records if record.is_positive)
    negatives = len(records) - positives
    if positives == 0:
        raise AssertionError(f"{split_name} split needs at least one positive image")
    if negatives == 0 and not allow_missing_negatives:
        raise AssertionError(f"{split_name} split needs at least one negative/background image")

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

    source_root = resolve_source_root(args, output_root)
    split_dirs, needs_split = find_split_dirs(source_root)

    all_records: list[ImageRecord] = []
    next_image_id = 1
    next_annotation_id = 1
    for split_name, image_dir, label_dir in split_dirs:
        records, next_image_id, next_annotation_id = read_records(
            image_dir=image_dir,
            label_dir=label_dir,
            output_split=split_name,
            target_class_id=args.target_class_id,
            start_image_id=next_image_id,
            start_annotation_id=next_annotation_id,
        )
        all_records.extend(records)

    if needs_split:
        train_records, val_records = split_records(all_records, args.val_fraction, args.seed)
    else:
        train_records = sorted([record for record in all_records if record.split == "train"], key=lambda record: record.id)
        val_records = sorted([record for record in all_records if record.split == "val"], key=lambda record: record.id)

    assert_dataset(train_records, "train", args.allow_missing_negatives)
    assert_dataset(val_records, "val", args.allow_missing_negatives)

    image_output_root = output_root / "images"
    copy_images(train_records + val_records, image_output_root)

    annotation_prefix = args.target_class_name.replace(" ", "_")
    write_coco_json(train_records, output_root / "annotations" / f"{annotation_prefix}_train.json", args.target_class_name)
    write_coco_json(val_records, output_root / "annotations" / f"{annotation_prefix}_val.json", args.target_class_name)

    print(
        "Prepared custom binary dataset: "
        f"{len(train_records)} train images, {len(val_records)} val images, "
        f"{sum(len(record.annotations) for record in train_records + val_records)} {args.target_class_name} boxes"
    )
    print(f"Source root: {source_root.resolve()}")
    print(f"Output root: {output_root.resolve()}")


if __name__ == "__main__":
    main()
