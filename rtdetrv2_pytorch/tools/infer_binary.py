"""Run inference for the binary RT-DETRv2 fine-tuned model.

This script is intentionally inference-only: it loads a trained checkpoint,
processes one image or a directory of images, and writes visualization plus JSON
prediction files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.core import YAMLConfig  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        default="configs/rtdetrv2/rtdetrv2_r18vd_binary_person_custom.yml",
        help="RT-DETRv2 config path",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default="output/rtdetrv2_r18vd_binary_person_custom/best.pth",
        help="trained checkpoint path",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="input image path or directory",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="output/inference_binary_person",
        help="directory for rendered images and JSON predictions",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda:0",
        help="inference device, for example cuda:0 or cpu",
    )
    parser.add_argument("--threshold", type=float, default=0.3, help="score threshold")
    parser.add_argument("--class-name", default="person", help="display name for class id 0")
    return parser.parse_args()


class DeployModel(nn.Module):
    def __init__(self, cfg: YAMLConfig) -> None:
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images: torch.Tensor, orig_target_sizes: torch.Tensor):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)


def list_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        images = sorted(path for path in input_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if images:
            return images
    raise FileNotFoundError(f"No input images found at {input_path}")


def load_model(config_path: str, checkpoint_path: str, device: torch.device) -> tuple[YAMLConfig, DeployModel]:
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)
    model = DeployModel(cfg).to(device).eval()
    return cfg, model


def draw_predictions(
    image: Image.Image,
    predictions: list[dict],
    output_path: Path,
    class_name: str,
) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for prediction in predictions:
        x1, y1, x2, y2 = prediction["box_xyxy"]
        score = prediction["score"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"{class_name} {score:.2f}", fill="blue")
    canvas.save(output_path)


def infer_one(
    model: DeployModel,
    cfg: YAMLConfig,
    image_path: Path,
    output_dir: Path,
    device: torch.device,
    threshold: float,
    class_name: str,
) -> None:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    eval_size = tuple(cfg.yaml_cfg.get("eval_spatial_size", [640, 640]))
    transform = T.Compose([T.Resize(eval_size), T.ToTensor()])
    image_tensor = transform(image)[None].to(device)
    orig_size = torch.tensor([[width, height]], device=device)

    with torch.no_grad():
        labels, boxes, scores = model(image_tensor, orig_size)

    keep = scores[0] >= threshold
    labels = labels[0][keep].detach().cpu()
    boxes = boxes[0][keep].detach().cpu()
    scores = scores[0][keep].detach().cpu()

    predictions = []
    for label, box, score in zip(labels.tolist(), boxes.tolist(), scores.tolist()):
        predictions.append(
            {
                "label": int(label),
                "class_name": class_name if int(label) == 0 else str(label),
                "score": float(score),
                "box_xyxy": [float(value) for value in box],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{image_path.stem}.json"
    image_output_path = output_dir / f"{image_path.stem}_pred.jpg"
    json_path.write_text(
        json.dumps(
            {
                "image": str(image_path),
                "width": width,
                "height": height,
                "threshold": threshold,
                "predictions": predictions,
            },
            indent=2,
        )
    )
    draw_predictions(image, predictions, image_output_path, class_name)
    print(f"{image_path}: {len(predictions)} detections -> {json_path}, {image_output_path}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg, model = load_model(args.config, args.resume, device)
    for image_path in list_images(Path(args.input)):
        infer_one(
            model=model,
            cfg=cfg,
            image_path=image_path,
            output_dir=Path(args.output_dir),
            device=device,
            threshold=args.threshold,
            class_name=args.class_name,
        )


if __name__ == "__main__":
    main()
