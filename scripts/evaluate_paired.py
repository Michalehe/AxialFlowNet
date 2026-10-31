"""Paired evaluation for AxialFlowNet.

For the paired mitochondrial data (``xunliti_paired``), ``trainA[i]`` and
``trainB[i]`` are the same slice before/after de-stretching, so quantitative
image-quality metrics are meaningful: run the generator on the A side and
compare against the B ground truth.

Reports PSNR / SSIM / L1 for a checkpoint, and (with ``--sweep_bound``) for
several ``texture_bound`` values, which is how the released
``inference_config.json`` values were chosen.

Usage
-----
::

    python scripts/evaluate_paired.py --ckpt checkpoints/xianliti_AxialFlowNet_200ep \
        --data datasets/xunlian-xianliti --phase train --limit 50

    # decide the architecture constants by re-running inference only
    python scripts/evaluate_paired.py --ckpt checkpoints/3axial_AxialFlowNet_100ep \
        --data datasets/3axial --phase test --sweep_bound 0.15 0.3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scripts.inference import build_generator, preprocess, resolve_checkpoint  # noqa: E402


def load_generator(ckpt: str | Path, texture_bound: float, device):
    ckpt_path = resolve_checkpoint(ckpt)
    net = build_generator(texture_bound=texture_bound).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.eval()
    return net


def pair_paths(data: Path, phase: str) -> list[tuple[Path, Path]]:
    a = sorted((data / f"{phase}A").glob("*.png"))
    b = sorted((data / f"{phase}B").glob("*.png"))
    if not a or not b:
        raise SystemExit(
            f"missing {phase}A / {phase}B under {data}\n"
            f"The released datasets ship only trainA/trainB plus an unpaired testA, "
            f"so use \"--phase train\" (or \"--limit N\" to cap the work)."
        )
    if len(a) != len(b):
        raise SystemExit(f"unpaired data: {len(a)} A vs {len(b)} B")
    return list(zip(a, b))


def evaluate(net, pairs, device, resize: str, load_size: int, limit: int) -> dict:
    n = min(len(pairs), limit) if limit else len(pairs)
    ps, ss, l1 = [], [], []
    for i, (pa, pb) in enumerate(pairs[:n]):
        img_a = Image.open(pa).convert("RGB")
        img_b = Image.open(pb).convert("RGB")
        if img_a.size != img_b.size:
            img_b = img_b.resize(img_a.size, Image.BICUBIC)
        tensor, _ = preprocess(img_a, resize, load_size)
        with torch.no_grad():
            fake, _ = net(tensor.to(device))
        from scripts.inference import tensor_to_image
        out = np.asarray(tensor_to_image(fake, img_a.size).convert("RGB"))
        ref = np.asarray(img_b.convert("RGB"))
        ps.append(psnr(ref, out, data_range=255))
        ss.append(ssim(ref, out, channel_axis=2, data_range=255))
        l1.append(np.abs(out.astype(np.int16) - ref.astype(np.int16)).mean() / 255.0)
        if (i + 1) % 25 == 0 or (i + 1) == n:
            print(f"    [{i + 1}/{n}]")
    return {"n": n, "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)),
            "l1": float(np.mean(l1)), "psnr_std": float(np.std(ps))}


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired evaluation for AxialFlowNet")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="dataset root with trainA/trainB (or testA/testB)")
    ap.add_argument("--phase", default="train", choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=0, help="max number of pairs (0 = all)")
    ap.add_argument("--resize", choices=["square", "native"], default="square")
    ap.add_argument("--load_size", type=int, default=256)
    ap.add_argument("--native_force_size", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--sweep_bound", type=float, nargs="*", default=None,
                    help="evaluate several texture_bound values")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    pairs = pair_paths(Path(args.data), args.phase)
    bounds = args.sweep_bound if args.sweep_bound else [0.15]
    load_size = args.load_size if args.resize == "square" else args.native_force_size

    print(f"{len(pairs)} pairs, mode={args.resize}, load_size={load_size}")
    rows = []
    for b in bounds:
        print(f"  texture_bound={b:g}")
        net = load_generator(args.ckpt, b, device)
        m = evaluate(net, pairs, device, args.resize, load_size, args.limit)
        rows.append((b, m))
        print(f"    PSNR {m['psnr']:.2f} dB  SSIM {m['ssim']:.4f}  L1 {m['l1']:.4f}  (n={m['n']})")
    if len(rows) > 1:
        best = max(rows, key=lambda r: r[1]["ssim"])
        print(f"\nbest texture_bound: {best[0]:g}  (SSIM {best[1]['ssim']:.4f}, PSNR {best[1]['psnr']:.2f} dB)")


if __name__ == "__main__":
    main()
