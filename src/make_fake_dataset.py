# ----------------------------------------------------------
# Make a tiny fake ImageFolder dataset for quick pipeline checks
# ----------------------------------------------------------
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_CLASSES = ["cats", "dogs"]


def make_image(w: int, h: int, color: tuple[int, int, int], label: str | None = None) -> Image.Image:
    img = Image.new("RGB", (w, h), color)
    if label:
        draw = ImageDraw.Draw(img)
        # Try to place text; if default font missing, just skip
        try:
            draw.text((10, 10), label, fill=(255, 255, 255))
        except Exception:
            pass
    return img


def class_color(idx: int) -> tuple[int, int, int]:
    # Deterministic distinct colors for first few classes
    palette = [
        (220, 20, 60),   # crimson
        (46, 139, 87),   # sea green
        (65, 105, 225),  # royal blue
        (255, 140, 0),   # dark orange
        (148, 0, 211),   # dark violet
    ]
    return palette[idx % len(palette)]


def generate_split(root: Path, split: str, classes: List[str], per_class: int, img_size: int) -> None:
    for ci, cls in enumerate(classes):
        cdir = root / split / cls
        cdir.mkdir(parents=True, exist_ok=True)
        color = class_color(ci)
        for i in range(per_class):
            img = make_image(img_size, img_size, color=color, label=f"{cls}")
            img.save(cdir / f"{cls}_{i:03d}.png")


def main():
    parser = argparse.ArgumentParser(description="Make a tiny fake ImageFolder dataset for testing")
    parser.add_argument("--root", type=str, default="data", help="Dataset root directory")
    parser.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES, help="Class names (default: cats dogs)")
    parser.add_argument("--train", type=int, default=20, help="Images per class for train")
    parser.add_argument("--val", type=int, default=10, help="Images per class for val")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    root = Path(args.root)
    generate_split(root, "train", args.classes, args.train, args.img_size)
    generate_split(root, "val", args.classes, args.val, args.img_size)

    print(f"Created fake dataset under {root}. Train/Val images per class: {args.train}/{args.val}")


if __name__ == "__main__":
    main()
