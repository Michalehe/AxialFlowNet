# Data documentation

This document describes the two microscopy datasets used in this work, the paired data loader, and
the preprocessing that inference must match.

Both datasets use the standard directory layout (`trainA`, `trainB`, `testA`), which the upstream
`unaligned` loader reads directly.

The image data itself is not distributed with this repository; this document records its structure,
format and preprocessing so the released weights can be applied to the corresponding acquisitions.

---

## 1. Axial leaf-tissue data

Unpaired axial leaf-tissue stretch. Domain A is the axially stretched material, domain B is the
de-stretched counterpart; **no per-image correspondence exists**, so this data is supervised only by
the adversarial and cycle-consistency objectives.

| Split | Files | Mode | Size distribution |
|---|---|---|---|
| `trainA` | 716 | `L` (8-bit greyscale) | 1024×382 (174), 256×100 (173), 512×249 (53), other mixed |
| `trainB` | 538 | `L` | 1024×400 (168), 512×249 (201), 421×277 (31) |
| `testA` | 165 | `L` | 421×277 (all) |

The split is genuinely unbalanced; **no `testB` is used**, because without a pairing it could not be
scored. The test split is for qualitative inspection.

Trained with `--preprocess resize_and_crop --load_size 286 --crop_size 256`, so images are square
resized and cropped. Inference must match:

```bash
python scripts/inference.py --ckpt checkpoints/3axial_AxialFlowNet_100ep \
    --input path/to/testA --output results/x --resize square
```

---

## 2. Paired mitochondrial serial sections

**Paired** mitochondrial serial sections. `trainA[i]` and `trainB[i]` are the *same* slice before and
after de-stretching, which makes quantitative image-quality metrics meaningful.

| Split | Files | Mode | Size |
|---|---|---|---|
| `trainA` | 566 | `I;16` (16-bit greyscale) | mixed native sizes, e.g. 512×162, 335×190 |
| `trainB` | 566 | `I;16` | matching native sizes |
| `testA` | 200 | `I;16` | 512×232 (all) |

Trained with `--preprocess none --dataset_mode xunliti_paired`: images are kept at native resolution
and reflect-padded to a multiple of 4, so **all resolution is preserved end to end**. Inference must
match, and the paired evaluation runs on the train split:

```bash
python scripts/inference.py --ckpt checkpoints/xianliti_AxialFlowNet_200ep \
    --input path/to/testA --output results/x --resize native
python scripts/evaluate_paired.py --ckpt checkpoints/xianliti_AxialFlowNet_200ep \
    --data path/to/paired_data --phase train --limit 120 --resize native
```

### 2.1 How `texture_bound` was determined for this run

`train_opt.txt` records every training flag **except** the two architecture constants
`--texture_bound` and `--flow_mag_threshold`, so they were recovered by re-running inference and
matching against archived reference outputs:

1. re-ran inference with both candidate bounds (0.15 and 0.3) on the 200-image `testA` split, in
   `native` mode, with `latest_net_G_A.pth` — mean abs pixel difference 0.334 (bound 0.15) vs 0.526
   (bound 0.3);
2. square resize was also tested and was far worse (mean abs difference 24.8), confirming that
   native mode is the correct preprocessing;
3. switching to `200_net_G_A.pth` sharpened the agreement to 0.037 (bound 0.15) vs 0.386 (bound
   0.30), a ~10× separation.

The residual 0.037/255 difference arises because the archived references were rendered from an
earlier export of the same filenames (see §4.2). The conclusion is unambiguous and is recorded in
`checkpoints/xianliti_AxialFlowNet_200ep/inference_config.json`.

---

## 3. The paired loader: `data/xunliti_paired_dataset.py`

The upstream `aligned` loader resizes both sides with the same transform, and the `unaligned` loader
randomly pairs them. Neither fits: the mitochondrial data has a genuine correspondence *and*
irregular native sizes. The paired loader therefore:

1. reads `trainA`/`testA` images at native resolution without any resize;
2. loads the matching `trainB` image, so A→B is a real ground-truth pair;
3. reflect-pads both sides to a multiple of 4 so the 4× down/up-sampling in the generator backbone
   divides evenly at every size;
4. normalises to `[-1, 1]` per channel, exactly as the upstream loader does.

Because no resize happens, `testA` (512×232) is fed at full resolution, and the generator's
`ConvTranspose2d` upsampling reconstructs it without the resampling artefacts a square resize
introduces.

---

## 4. Reproduction notes

1. **The two checkpoints use different preprocessing.** The leaf-tissue run was trained square
   resized and must be run with `--resize square`; `xianliti_AxialFlowNet_200ep` was trained native
   and must be run with `--resize native`. Feeding one the other's mode changes the resolution the
   generator sees and degrades the result substantially — square mode on the native-resolution
   mitochondrial data is measurably worse (mean abs pixel difference of 24.8 versus 0.34 against the
   stored reference outputs).

2. **Stored reference images came from a different source generation.** The archived reference
   results were produced from an earlier 512×232 export of the same filenames. Re-running with the
   released code on the current data reproduces them closely but not bit-exactly (mean abs
   difference 0.037 of 255 per pixel, 2.8 % of pixels differing), whereas selecting the *wrong*
   `texture_bound` produces a much larger difference (0.386).

3. **`train_opt.txt` in each checkpoint directory is authoritative.** It records the exact command
   line used, including `preprocess`, `load_size`, `dataset_mode` and `gan_mode`.

4. **Metrics are not comparable across the two datasets.** The paired mitochondria data yields real
   PSNR/SSIM; the unpaired leaf-tissue data has no per-pixel ground truth, so only qualitative or
   FID-style comparison is possible.
