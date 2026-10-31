"""Build the stacked example panels used in README.md.

Each panel puts the raw input, the de-stretched output and the predicted flow
field (when available) into one figure. Run this after ``scripts/inference.py``
has produced ``examples/<tag>`` directories.

Usage::

    python scripts/make_examples.py --root examples --output examples/panels
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def stack(input_path: Path, out_path: Path, flow_path: Path | None, panel: Path, gap: int = 6) -> None:
    input_img = Image.open(input_path).convert("RGB")
    out_img = Image.open(out_path).convert("RGB")
    imgs = [("input", input_img), ("output", out_img)]
    if flow_path is not None and flow_path.exists():
        imgs.append(("flow", Image.open(flow_path).convert("RGB")))

    height = max(im.height for _, im in imgs)
    labels = []
    for name, im in imgs:
        scale = height / im.height
        labels.append((im.resize((max(1, int(round(im.width * scale))), height), Image.BICUBIC)))
    total_w = sum(im.width for im in labels) + gap * (len(labels) - 1)
    canvas = Image.new("RGB", (total_w, height + 24), (255, 255, 255))
    x = 0
    draw = ImageDraw.Draw(canvas)
    for (name, _), im in zip(imgs, labels):
        canvas.paste(im, (x, 24))
        draw.rectangle([x, 0, x + im.width, 20], fill=(245, 245, 245))
        draw.text((x + 4, 4), name, fill=(40, 40, 40))
        x += im.width + gap
    panel.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(panel)
    print(f"  {panel.name}  ({canvas.width}x{canvas.height})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build example panels")
    ap.add_argument("--root", default="examples", help="directory holding examples/<tag>/...")
    ap.add_argument("--output", default="examples/panels", help="destination directory")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output)
    for tag_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for out_png in sorted(tag_dir.glob("*.png")):
            if out_png.name.endswith("_flow.png") or tag_dir.name == "panels":
                continue
            stem = out_png.stem
            # locate the matching raw input by name; prefer the test split
            candidates = list(Path("datasets").rglob(stem + ".png"))
            if not candidates:
                print(f"  skip {stem}: raw input not found under datasets/")
                continue
            raw_input = next((p for p in candidates if p.parent.name.endswith("testA")), candidates[0])
            flow_png = tag_dir / f"{stem}_flow.png"
            stack(raw_input, out_png, flow_png, out_dir / f"{tag_dir.name}__{stem}.png")
    print(f"panels written to {out_dir}")


if __name__ == "__main__":
    main()
