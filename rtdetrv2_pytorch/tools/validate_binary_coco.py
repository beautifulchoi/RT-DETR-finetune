"""Validate binary COCO annotations before RT-DETRv2 fine-tuning."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ann_files", nargs="+", type=Path, help="COCO annotation JSON files")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--expected-category-id", default=0, type=int)
    parser.add_argument("--expected-category-name", default="person")
    parser.add_argument("--allow-missing-negatives", action="store_true")
    return parser.parse_args()


def validate_file(
    ann_file: Path,
    num_classes: int,
    expected_category_id: int,
    expected_category_name: str,
    allow_missing_negatives: bool,
) -> None:
    data = json.loads(ann_file.read_text())
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    errors: list[str] = []
    image_ids = {image.get("id") for image in images}
    category_ids = [category.get("id") for category in categories]
    category_names = [category.get("name") for category in categories]
    annotation_category_ids = [annotation.get("category_id") for annotation in annotations]

    if categories != [{"id": expected_category_id, "name": expected_category_name, "supercategory": "object"}]:
        errors.append(
            "categories must be exactly "
            f"[{{'id': {expected_category_id}, 'name': '{expected_category_name}', 'supercategory': 'object'}}], "
            f"got ids={category_ids}, names={category_names}"
        )

    for annotation in annotations:
        category_id = annotation.get("category_id")
        if not isinstance(category_id, int) or not 0 <= category_id < num_classes:
            errors.append(
                f"annotation id={annotation.get('id')} has category_id={category_id}; "
                f"for num_classes={num_classes}, valid ids are 0..{num_classes - 1}"
            )

        if annotation.get("image_id") not in image_ids:
            errors.append(f"annotation id={annotation.get('id')} references missing image_id={annotation.get('image_id')}")

        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"annotation id={annotation.get('id')} has invalid bbox={bbox}")
        elif bbox[2] <= 0 or bbox[3] <= 0:
            errors.append(f"annotation id={annotation.get('id')} has non-positive bbox width/height={bbox}")

    positive_image_ids = {annotation.get("image_id") for annotation in annotations}
    negative_count = len(image_ids - positive_image_ids)
    if positive_image_ids and negative_count == 0 and not allow_missing_negatives:
        errors.append("no negative/background images found")
    if not positive_image_ids:
        errors.append("no positive images found")

    print(f"{ann_file}:")
    print(f"  images={len(images)} annotations={len(annotations)}")
    print(f"  annotation category ids={dict(Counter(annotation_category_ids))}")
    print(f"  positive_images={len(positive_image_ids)} negative_images={negative_count}")

    if errors:
        print("  FAILED")
        for error in errors[:20]:
            print(f"  - {error}")
        if len(errors) > 20:
            print(f"  - ... {len(errors) - 20} more errors")
        raise SystemExit(1)

    print("  OK")


def main() -> None:
    args = parse_args()
    for ann_file in args.ann_files:
        validate_file(
            ann_file=ann_file,
            num_classes=args.num_classes,
            expected_category_id=args.expected_category_id,
            expected_category_name=args.expected_category_name,
            allow_missing_negatives=args.allow_missing_negatives,
        )


if __name__ == "__main__":
    main()
