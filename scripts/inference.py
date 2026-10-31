"""AxialFlowNet 2D inference.

Runs the released geometric generator (flow branch + texture residual branch)
on 2D images and writes the de-stretched results as PNG.

The generator is ``_GeometricGenerator`` (see ``models/axialflownet_model.py``):
it returns ``(fake, flow)`` where ``fake = clamp(warp(x, flow) + texture, -1, 1)``.

Examples
--------
Single image::

    python scripts/inference.py --ckpt checkpoints/3axial_AxialFlowNet_100ep \
        --input mydata/input020000.png --output results/demo

Whole folder::

    python scripts/inference.py --ckpt checkpoints/3axial_AxialFlowNet_100ep \
        --input mydata/testA --output results/testA --resize square

Checkpoints come with an ``inference_config.json`` recording the architecture
constants used while training that run; they are applied automatically when
``--ckpt`` points at a run directory (or at a weights file placed next to one).
Both can be overridden explicitly on the command line.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from models import networks  # noqa: E402
from models.axialflownet_model import _GeometricGenerator  # noqa: E402

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Architecture constants of the released weights. These only matter at
# inference time for the generator's texture branch (`--texture_bound`).
# The flow-magnitude threshold is a training-time loss constant only.
DEFAULT_TEXTURE_BOUND = 0.15
CONFIG_NAME = "inference_config.json"


def build_generator(texture_bound: float = DEFAULT_TEXTURE_BOUND) -> "_GeometricGenerator":
    """Instantiate the released generator architecture."""
    norm_layer = networks.get_norm_layer("instance")
    return _GeometricGenerator(
        input_nc=3,
        output_nc=3,
        ngf=64,
        norm_layer=norm_layer,
        use_dropout=False,
        n_blocks=9,
        texture_bound=texture_bound,
    )


def resolve_checkpoint(ckpt: str | Path) -> Path:
    """Accept either a weights file or a checkpoint run directory."""
    p = Path(ckpt)
    if p.is_dir():
        for name in ("latest_net_G_A.pth", "100_net_G_A.pth"):
            if (p / name).exists():
                return p / name
        cands = sorted(p.glob("*_net_G_A.pth"))
        if not cands:
            raise FileNotFoundError(f"no *_net_G_A.pth in {p}")
        return cands[-1]
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    return p


def load_config(ckpt_path: Path) -> dict:
    """Read inference_config.json from the checkpoint directory, if present."""
    cfg_path = ckpt_path.parent / CONFIG_NAME
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ignoring malformed {cfg_path}: {exc}")
    return {}


def load_generator(ckpt: str | Path, texture_bound: float | None = None, device: torch.device | None = None):
    """Build the generator and load weights. Returns (net, config, ckpt_path)."""
    device = device or _default_device()
    ckpt_path = resolve_checkpoint(ckpt)
    cfg = load_config(ckpt_path)
    if texture_bound is None:
        texture_bound = float(cfg.get("texture_bound", DEFAULT_TEXTURE_BOUND))
    net = build_generator(texture_bound=texture_bound).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}
    net.load_state_dict(state)
    net.eval()
    print(f"  loaded {ckpt_path}  texture_bound={texture_bound:g}")
    return net, cfg, ckpt_path


def _default_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _pad_to_multiple(img: Image.Image, multiple: int = 4, mode: str = "reflect") -> tuple[Image.Image, int, int]:
    """Reflect-pad so H and W are multiples of <multiple>; returns (img, H, W)."""
    w, max_h = img.size  # PIL: (W, H)
    new_w = ((w + multiple - 1) // multiple) * multiple
    new_h = ((max_h + multiple - 1) // multiple) * multiple
    pad_w, pad_h = new_w - w, new_h - max_h
    if pad_w or pad_h:
        pl, pr = pad_w // 2, pad_w - pad_w // 2
        pt, pb = pad_h // 2, pad_h - pad_h // 2
        img = transforms.functional.pad(img, (pl, pt, pr, pb), padding_mode=mode)
    return img, max_h, w


def preprocess(img: Image.Image, resize: str, load_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Convert a PIL image to a normalised tensor.

    ``resize == "square"`` resizes to load_size x load_size (matches the
    released training configuration ``--preprocess resize_and_crop`` used for
    the 3-axial data). ``resize == "native"`` keeps the original resolution and
    reflect-pads to a multiple of 4 (matches ``--preprocess none`` used for the
    paired mitochondrial data).

    Returns (tensor, original (W, H) to restore afterwards).
    """
    w, h = img.size
    if resize == "square":
        resized = img.resize((load_size, load_size), Image.BICUBIC)
    else:
        if load_size and load_size > 0:
            # force the long side, then pad to a square of load_size
            if w >= h:
                new_w, new_h = load_size, max(1, int(round(h * load_size / w)))
            else:
                new_h, new_w = load_size, max(1, int(round(w * load_size / h)))
            resized = img.resize((new_w, new_h), Image.BICUBIC)
            pad_w, pad_h = load_size - new_w, load_size - new_h
            pl, pr = pad_w // 2, pad_w - pad_w // 2
            pt, pb = pad_h // 2, pad_h - pad_h // 2
            if pad_w or pad_h:
                resized = transforms.functional.pad(resized, (pl, pt, pr, pb), padding_mode="reflect")
        else:
            resized, _, _ = _pad_to_multiple(img, 4, "reflect")
    tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])(resized).unsqueeze(0)
    return tensor, (w, h)


def tensor_to_image(tensor: torch.Tensor, out_size: tuple[int, int] | None = None) -> Image.Image:
    """Map a network output in [-1, 1] back to a uint8 PIL image."""
    t = tensor.squeeze(0).detach().cpu().float()
    t = (t * 0.5) + 0.5
    t = t.clamp(0, 1)
    img = transforms.ToPILImage()(t)
    if out_size is not None:
        img = img.resize(out_size, Image.BICUBIC)
    return img


def generate(net, image_path: Path, resize: str, load_size: int, device) -> tuple[Image.Image, torch.Tensor, tuple[int, int]]:
    """Run the generator on one image. Returns (output image, flow tensor, original size)."""
    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    tensor, _ = preprocess(img, resize, load_size)
    with torch.no_grad():
        fake, flow = net(tensor.to(device))
    return tensor_to_image(fake, original_size), flow, original_size


def _make_color_wheel() -> list[tuple[float, float, float]]:
    """Middlebury-style optical-flow colour wheel (direction -> hue)."""
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    wheel = [(0.0, 0.0, 0.0)] if ncols == 255 else []
    for i in range(RY):
        wheel.append((255.0, 255.0 * i / RY, 0.0))
    for i in range(YG):
        wheel.append((255.0 - 255.0 * i / YG, 255.0, 0.0))
    for i in range(GC):
        wheel.append((0.0, 255.0, 255.0 * i / GC))
    for i in range(CB):
        wheel.append((0.0, 255.0 - 255.0 * i / CB, 255.0))
    for i in range(BM):
        wheel.append((255.0 * i / BM, 0.0, 255.0))
    for i in range(MR):
        wheel.append((255.0, 0.0, 255.0 - 255.0 * i / MR))
    return wheel


_COLOR_WHEEL = _make_color_wheel()


def flow_to_image(flow: torch.Tensor, scale: float | None = None, max_mag: float | None = None) -> Image.Image:
    """Colour-code a flow field for inspection (Middlebury convention).

    Direction maps to hue, magnitude to saturation; a zero flow renders as white.
    ``scale`` is kept for backwards compatibility and is interpreted as ``max_mag``
    when ``max_mag`` is not given; when both are None the field's own maximum is used
    so the picture is never saturated.
    """
    if max_mag is None:
        max_mag = scale
    f = flow.squeeze(0).detach().cpu().float()
    fx, fy = f[0], f[1]
    mag = torch.sqrt(fx * fx + fy * fy + 1e-12)
    if max_mag is None or not (max_mag > 0):
        max_mag = float(mag.max()) or 1.0

    n = len(_COLOR_WHEEL)
    angle = (torch.atan2(fy, fx) + math.pi) / (2 * math.pi) * n
    wheel = torch.tensor(_COLOR_WHEEL, dtype=torch.float32).T  # (3, n)

    one_minus = (mag / max_mag).clamp(0, 1)

    # linear interpolation between the two neighbouring colour-wheel bins
    base = angle.floor()
    t = angle - base
    ia = base.to(torch.long)
    ib = (ia + 1) % n
    r = torch.lerp(wheel[0][ia], wheel[0][ib], t)
    g = torch.lerp(wheel[1][ia], wheel[1][ib], t)
    b = torch.lerp(wheel[2][ia], wheel[2][ib], t)
    rgb = torch.stack([r, g, b], dim=0) / 255.0
    # fade to white as the displacement magnitude drops
    rgb = rgb * one_minus.unsqueeze(0) + (1.0 - one_minus).unsqueeze(0)
    return transforms.ToPILImage()((rgb * 255).byte())


def collect_inputs(inp: Path) -> list[Path]:
    if inp.is_file():
        return [inp]
    return sorted(p for p in inp.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="AxialFlowNet 2D inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples")[-1],
    )
    ap.add_argument("--ckpt", required=True,
                    help="generator weights (*latest_net_G_A.pth*) or a checkpoint run directory")
    ap.add_argument("--input", required=True, help="input image file or directory")
    ap.add_argument("--output", required=True, help="output directory")
    ap.add_argument("--resize", choices=["square", "native"], default="square",
                    help="square: resize to --load_size (3-axial runs); native: keep resolution, "
                         "pad to multiple of 4 (paired runs)")
    ap.add_argument("--load_size", type=int, default=256, help="input size for --resize square")
    ap.add_argument("--native_force_size", type=int, default=0,
                    help="force the long side to this size in --resize native (0 = no resize)")
    ap.add_argument("--texture_bound", type=float, default=None,
                    help="override --texture_bound used while training (default: read from "
                         "inference_config.json, else %.2f)" % DEFAULT_TEXTURE_BOUND)
    ap.add_argument("--save_flow", action="store_true", help="also write the predicted flow field")
    ap.add_argument("--suffix", default="", help="append this to output file names")
    ap.add_argument("--device", default=None, help="cuda:0 / cuda / cpu (default: auto)")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else _default_device()
    print("device:", device)
    net, cfg, ckpt_path = load_generator(args.ckpt, args.texture_bound, device)

    in_path = Path(args.input)
    files = collect_inputs(in_path)
    if not files:
        raise SystemExit(f"no images found in {in_path}")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_size = args.load_size if args.resize == "square" else args.native_force_size
    print(f"mode={args.resize} load_size={load_size}  files={len(files)}")
    for f in files:
        out_img, flow, orig = generate(net, f, args.resize, load_size or 256, device)
        out_name = f.stem + ("_" + args.suffix if args.suffix else "") + ".png"
        out_img.save(out_dir / out_name)
        if args.save_flow:
            # colour-code the flow with the field's own maximum so nothing saturates
            flow_to_image(flow).save(out_dir / f"{f.stem}_flow.png")
        print(f"  {f.name} ({orig[0]}x{orig[1]}) -> {out_name}")
    print(f"\nsaved {len(files)} image(s) to {out_dir}")


if __name__ == "__main__":
    main()
