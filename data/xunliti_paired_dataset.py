"""Paired dataset that preserves original image content for CycleGAN training.

Why this exists:
    The default UnalignedDataset applies `resize_and_crop` (resize to 286x286 then random
    crop to 256x256). For paired xy/xz slices, this:
      - Breaks pixel-level correspondence between A and B.
      - Destroys the original aspect ratio (and thus the geometric stretch axis info).

This dataset:
    1. Pairs A[idx] with B[idx] by index (1-to-1 correspondence, no random pairing).
    2. Does NOT resize or random-crop. Pads each image independently with REFLECT padding
       to the next multiple of 4 (so CycleGAN's two downsamples by 2 still work).
       Both A and B keep their original aspect ratio and pixel content.
    3. No random flipping (paired data — flipping would break xy/xz correspondence).
"""
import os
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms


def _pad_to_multiple(img, multiple=4, mode="reflect"):
    """Reflect-pad an image so H and W are both multiples of <multiple>.
    Padding is symmetric on each axis when possible, with the extra 1px on the bottom/right."""
    w, h = img.size  # PIL: (W, H)
    new_w = ((w + multiple - 1) // multiple) * multiple
    new_h = ((h + multiple - 1) // multiple) * multiple
    pad_w = new_w - w
    pad_h = new_h - h
    if pad_w == 0 and pad_h == 0:
        return img
    # split padding: left/top take floor, right/bottom take ceil
    pl = pad_w // 2
    pr = pad_w - pl
    pt = pad_h // 2
    pb = pad_h - pt
    return transforms.functional.pad(img, (pl, pt, pr, pb), padding_mode=mode)


class XunlitiPairedDataset(BaseDataset):
    """CycleGAN-style paired dataset, but preserves original aspect ratio via padding."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(preprocess="none")
        parser.add_argument("--pad_multiple", type=int, default=4,
                            help="pad H/W to next multiple of this number")
        parser.add_argument("--pad_mode", type=str, default="reflect",
                            choices=["reflect", "replicate", "constant"],
                            help="padding mode for the short side")
        parser.add_argument("--force_size", type=int, default=0,
                            help="if >0, force a square of this size (pad with reflection)")
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        if len(self.A_paths) != len(self.B_paths):
            raise ValueError(
                f"Paired dataset requires equal counts. Got A={len(self.A_paths)} B={len(self.B_paths)}"
            )
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.size = self.A_size
        self.pad_multiple = opt.pad_multiple
        self.pad_mode = opt.pad_mode
        self.force_size = opt.force_size

    def __len__(self):
        return self.size

    def _process(self, path):
        img = Image.open(path).convert("RGB")
        if self.force_size > 0:
            # Resize so the LONG side equals force_size, then pad to square.
            w, h = img.size
            if w >= h:
                new_w = self.force_size
                new_h = max(1, int(round(h * new_w / w)))
            else:
                new_h = self.force_size
                new_w = max(1, int(round(w * new_h / h)))
            img = img.resize((new_w, new_h), Image.BICUBIC)
            # Now pad to square
            pad_w = self.force_size - new_w
            pad_h = self.force_size - new_h
            pl = pad_w // 2
            pr = pad_w - pl
            pt = pad_h // 2
            pb = pad_h - pt
            img = transforms.functional.pad(img, (pl, pt, pr, pb), padding_mode=self.pad_mode)
        else:
            img = _pad_to_multiple(img, self.pad_multiple, self.pad_mode)
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return tf(img)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        A = self._process(A_path)
        B = self._process(B_path)
        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}