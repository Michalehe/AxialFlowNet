"""AxialFlowNet 3D de-stretch for microscopy stacks.

Applies the released 2D generator slice by slice to a ZYX .tif stack, which is
the configuration used for the axial nuclear / mitochondrial data.

The generator is geometry-only safe to run per slice: it predicts a per-slice
flow field, so there is no cross-slice smoothing and no z-axis assumptions.

Usage
-----
::

    python scripts/destretch3d.py --ckpt checkpoints/xianliti_AxialFlowNet_200ep \
        --input data/volume.tif --output results/destretch3d

Multiple inputs may be given (files or directories).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import tifffile
from PIL import Image

PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scripts.inference import load_generator, preprocess, tensor_to_image  # noqa: E402


def process_slice(arr: np.ndarray, net, device, resize: str, load_size: int) -> np.ndarray:
    """De-stretch one (H=Y, W=X) slice; returns uint8 of the same shape."""
    img = Image.fromarray(arr).convert("RGB")
    tensor, _ = preprocess(img, resize, load_size or 256)
    with torch.no_grad():
        fake, _ = net(tensor.to(device))
    return np.asarray(tensor_to_image(fake, img.size).convert("L"), dtype=np.uint8)


def iter_inputs(paths) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.extend(sorted(q for q in p.rglob("*") if q.suffix.lower() in (".tif", ".tiff")))
        else:
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="AxialFlowNet 3D de-stretch")
    ap.add_argument("--ckpt", required=True, help="generator weights or checkpoint run directory")
    ap.add_argument("--input", nargs="+", required=True, help=".tif volumes, or directories of them")
    ap.add_argument("--output", required=True, help="output directory")
    ap.add_argument("--resize", choices=["square", "native"], default="square")
    ap.add_argument("--load_size", type=int, default=256)
    ap.add_argument("--native_force_size", type=int, default=0)
    ap.add_argument("--texture_bound", type=float, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    net, _, _ = load_generator(args.ckpt, args.texture_bound, device)
    load_size = args.load_size if args.resize == "square" else args.native_force_size

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = iter_inputs(args.input)
    if not files:
        raise SystemExit("no .tif inputs found")

    for src in files:
        vol = tifffile.imread(src)
        if vol.ndim == 2:
            vol = vol[None, ...]
        if vol.ndim != 3:
            raise SystemExit(f"{src}: expected a 3D ZYX stack, got shape {vol.shape}")
        z, y, x = vol.shape
        out = np.empty((z, y, x), dtype=np.uint8)
        print(f"\n{src.name}: input {vol.shape} {vol.dtype}")
        for i in range(z):
            out[i] = process_slice(np.asarray(vol[i]), net, device, args.resize, load_size)
            if (i + 1) % 50 == 0 or (i + 1) == z:
                print(f"  [{i + 1}/{z}]")
        dst = out_dir / f"{src.stem}_destretched.tif"
        tifffile.imwrite(dst, out, photometric="minisblack")
        print(f"  -> {dst}")


if __name__ == "__main__":
    main()
