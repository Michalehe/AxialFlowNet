# Methods

Description of AxialFlowNet as implemented in `models/axialflownet_model.py`. Every statement here
corresponds to the released code.

---

## 1. Overview

The task is one-directional: axially stretched serial sections need a geometric correction, and the
correction is a *spatial* deformation rather than a global intensity remapping. AxialFlowNet keeps
the bidirectional adversarial framework and splits the generator output into two cooperating
branches fed by a shared backbone:

- a **flow branch** predicting a 2-D displacement field `u`, used to warp the *input* image;
- a **texture-residual branch** predicting a bounded pixel correction `T`.

The generator output is therefore

\[
\hat I \;=\; \operatorname{Clamp}\!\left[\,\mathcal{W}(I,\,\mathbf{u}) \;+\; T\,\right]_{-1,\,1},
\]

so the final image is the warped input plus a bounded residual. Spatial deformation and intensity
correction are produced by separate paths and can be constrained and inspected independently.

---

## 2. Generator: `_GeometricGenerator`

### 2.1 Backbone

Shared feature extraction, `ngf` channels at the input (64 for all released weights):

```
input -> ReflectionPad(3) -> Conv 7x7 -> InstanceNorm -> ReLU
      -> 2 x [Conv 3x3 stride 2 -> Norm -> ReLU]        # 1/2, then 1/4 resolution
      -> n_blocks x ResnetBlock (9 by default)
      -> 2 x [ConvTranspose2d 3x3 stride 2 -> Norm -> ReLU]
      -> features F  (ngf channels, full resolution)
```

Two stride-2 convolutions reduce the spatial size by 4 and two transposed convolutions restore it.
The same backbone serves both branches, so geometry and appearance are predicted from one shared
feature representation.

### 2.2 Flow branch (`flow_head.py`)

`FlowHead` consumes `F` (64 channels) and emits the **2-channel** displacement field
`u(x,y) = (u_x, u_y)`. It is three 3×3 convolutions with `LeakyReLU(0.2)` between them and a final
`tanh`, so the flow is bounded to `[-1, 1]`.

The final convolution is **zero-initialised** (weight *and* bias), so at the start of training
`u ≡ 0` exactly and the warp is the identity. The generator begins as a pure texture corrector and
learns deformation progressively.

### 2.3 Texture-residual branch

```
ReflectionPad(3) -> Conv 7x7  (ngf -> output_nc) -> Tanh, scaled by texture_bound
```

so `T ∈ [-texture_bound, texture_bound]`. The bound keeps this branch supplementary: it corrects
intensity and local texture, and cannot on its own rewrite an image.

This scaling constant is part of the architecture, so `--texture_bound` must be set to the value used
at training time when reloading a checkpoint. The released runs use 0.3 (`3axial_AxialFlowNet_100ep`)
and 0.15 (`xianliti_AxialFlowNet_200ep`); each run directory carries an `inference_config.json`
recording its own value.

### 2.4 Warping and fusion

`warp_with_flow(input, flow)` builds the identity sampling grid, offsets it by the displacement
field, and calls `grid_sample` (`mode="bilinear"`, `padding_mode="border"`, `align_corners=True`).
The deformation is differentiable and requires no optical-flow supervision.

The field is mapped into normalised grid coordinates as

\[
u_x \mapsto \frac{2\,u_x}{W},\qquad u_y \mapsto \frac{2\,u_y}{H},
\]

then added to the grid in `[-1, 1]`. One normalised step spans one pixel over `W`/`H` pixels, so
**a flow component of value `v` displaces image content by exactly `v` pixels**. Warping a block of
columns with a uniform `u_x = 1.0` shifts the block by precisely 1 pixel (measured centroid
−1.00 px).

Two consequences:

- the `tanh` bound `[-1, 1]` is a **one-pixel** bound;
- `flow_mag_threshold` (0.3 or 0.4) is a **sub-pixel** soft cap: deviations under ~0.3 px are free,
  and only larger local deformations are penalised.

Outputs of `forward()` are `(fake, flow)`, which the inference scripts expose directly.

---

## 3. Discriminator: `AxialMultiscaleDiscriminator`

Two concurrent branches, both wrapped in **spectral norm**, whose outputs are concatenated along the
channel axis and fused by two 1×1 convolutions into a single 1-channel prediction map:

| Branch | Weights | Purpose |
|---|---|---|
| `_SpatialPatchBranch` | 70×70 PatchGAN: conv+`LeakyReLU(0.2)`, stride 2, then two stride-2 layers and one stride-1 layer | local 2-D texture |
| `_AxialStripBranch` | conv stack with kernel `(3, 8)`, stride `(1, 2)`, padding `(1, 3)` | elongated co-occurrence along the image width |

The axial branch suits stretched sectioning material: an elongated kernel spans several pixels along
the axis carrying the stretch, so the discriminator can penalise the residual directionality that an
isotropic patch statistic would average away. If the two branch outputs differ in size, the axial
output is bilinearly resized to the spatial one before fusion.

Both branches see the same input resolution; there is no image pyramid. The architecture is a
two-branch spatial + elongated-kernel discriminator, and the manuscript should describe it that way.

### 3.1 Which discriminator sees which domain

The model instantiates two of them, and the naming follows the **data**:

| Instance | Compares |
|---|---|
| `netD_A` | real `B` images vs. generated `B` images |
| `netD_B` | real `A` images vs. generated `A` images |

`netD_A` is the discriminator for the B domain, i.e. the one trained on `G_A`'s output. The
architecture diagram should follow this data flow.

---

## 4. Losses

The two released runs share the GAN, cycle and structural weights. The loss weights are read from
each run's `train_opt.txt`; `texture_bound` and `flow_mag_threshold` come from each run's
`inference_config.json`:

| Weight | `3axial_AxialFlowNet_100ep` | `xianliti_AxialFlowNet_200ep` |
|---|---|---|
| `lambda_A` / `lambda_B` | 10 / 10 | 10 / 10 |
| `lambda_identity` | 0.5 | 0.5 |
| `lambda_struct` | 1.0 | 1.0 |
| `struct_ssim_w` / `struct_edge_w` | 1.0 / 0.5 | 1.0 / 0.5 |
| `lambda_mono` | 2.0 | 2.0 |
| `lambda_flow_smooth` | 2.0 | 2.0 |
| `lambda_flow_mono` | 5.0 | 5.0 |
| `lambda_flow_mag` | 1.0 | 1.0 |
| `texture_bound` | 0.3 | 0.15 |
| `flow_mag_threshold` | 0.3 | 0.4 |
| `gan_mode` | vanilla | vanilla |
| `ngf` / `ndf` / `norm` | 64 / 64 / instance | 64 / 64 / instance |

`alpha_SSIM = 1.0` and `alpha_edge = 0.5` are the structure-loss internals (`--struct_ssim_w`,
`--struct_edge_w`).

### 4.1 Adversarial and cycle-consistency

Standard CycleGAN terms with `gan_mode = vanilla` (LSGAN) for both released runs. Cycle consistency
is L1 in both directions, inside the structural term (§4.2).

### 4.2 Structural loss

\[
\mathcal{L}_{\mathrm{struct}}
= \underbrace{\|\tilde I - I\|_1}_{\text{pixel}}
+ \alpha_{\mathrm{SSIM}}\,(1-\mathrm{SSIM})
+ \alpha_{\mathrm{edge}}\,\underbrace{\|\nabla \tilde I - \nabla I\|_1}_{\text{Sobel}},
\qquad
\alpha_{\mathrm{SSIM}}=1.0,\;\; \alpha_{\mathrm{edge}}=0.5 .
\]

Pixel error alone rewards blurry reconstructions; the SSIM term constrains local structure and
contrast, and the Sobel term compares edges explicitly, which is what survives a sectioning artefact
and what a de-stretching correction must not destroy.

### 4.3 Axial monotonicity

Applied to the *reconstruction error* `E = \tilde I - I` along the image height:

\[
\mathcal{L}_{\mathrm{mono}}
= \operatorname{mean}\!\left[\operatorname{ReLU}\!\left(-(E_{y+1}-E_y)\right)\right].
\]

Only negative row-to-row differences are penalised. This constrains how the error varies vertically,
with the image y-axis aligned to the physical axis along which the sections are stacked.

### 4.4 Flow regularisation

\[
\mathcal{L}_{\mathrm{flow\_smooth}}
= \tfrac{1}{2}\left(\operatorname{mean}|\Delta_x \mathbf{u}| + \operatorname{mean}|\Delta_y \mathbf{u}|\right)
\]

a total-variation prior that penalises local folds and speckle while permitting smooth global
displacement.

\[
\mathcal{L}_{\mathrm{flow\_mono}}
= \operatorname{mean}\!\left[\operatorname{ReLU}\!\left(-\Delta_y u_y\right)\right]
\]

penalises decreases of the vertical flow component along the vertical axis, encoding the prior that
the axial correction varies monotonically along the stack direction.

\[
m = \sqrt{u_x^2+u_y^2+\varepsilon},\qquad
\mathcal{L}_{\mathrm{flow\_mag}} = \operatorname{mean}\!\left[\operatorname{ReLU}(m-\tau)\right]
\]

a soft hinge capping displacement magnitude, with `τ = flow_mag_threshold`. Displacements below τ
cost nothing, so small corrections are free and only large deformations are penalised. Because of
the scaling established in §2.4, **τ is measured in pixels** (0.3 px and 0.4 px for the two released
runs).

### 4.5 Identity

With `lambda_identity > 0`, `G_A(I_B)` is forced back toward `I_B` (and symmetrically for `G_B`),
which limits gratuitous recolouring when the input is already in the target domain. Weight 0.5.

### 4.6 Total objective

As summed in `backward_G()`:

\[
\begin{aligned}
\mathcal{L}_G ={}&
\underbrace{\mathcal{L}_{\mathrm{GAN}}^{A\to B}+\mathcal{L}_{\mathrm{GAN}}^{B\to A}}_{\mathcal{L}_{\mathrm{adv}}}
+ \lambda_{\mathrm{struct}}\Big[\lambda_A\ \mathcal{L}_{\mathrm{struct}}^{A} + \lambda_B\ \mathcal{L}_{\mathrm{struct}}^{B}\Big] \\
&+ \lambda_{\mathrm{mono}}\Big[\mathcal{L}_{\mathrm{mono}}^{A} + \mathcal{L}_{\mathrm{mono}}^{B}\Big]
+ \lambda_{\mathrm{fs}}\cdot\tfrac12\big[\mathcal{L}_{\mathrm{fs}}^{A\!B}+\mathcal{L}_{\mathrm{fs}}^{B\!A}\big] \\
&+ \lambda_{\mathrm{fm}}\cdot\tfrac12\big[\mathcal{L}_{\mathrm{fm}}^{A\!B}+\mathcal{L}_{\mathrm{fm}}^{B\!A}\big]
+ \lambda_{\mathrm{fmag}}\cdot\tfrac12\big[\mathcal{L}_{\mathrm{fmag}}^{A\!B}+\mathcal{L}_{\mathrm{fmag}}^{B\!A}\big] \\
&+ \underbrace{\lambda_B\lambda_{\mathrm{id}}\,\|\mathrm{idt}_A-I_B\|_1
+ \lambda_A\lambda_{\mathrm{id}}\,\|\mathrm{idt}_B-I_A\|_1}_{\mathcal{L}_{\mathrm{id}}} .
\end{aligned}
\]

The two structural terms carry `lambda_struct` *and* the directional `lambda_A`/`lambda_B`, whereas
the monotonicity and flow terms carry only their own weight. The reconstruction L1 enters through
`L_struct`; `loss_cycle_A`/`loss_cycle_B` are computed for monitoring and are not added to `loss_G`.

### 4.7 Training procedure

Alternating updates. The generator step computes both directions, the cycle, the structural,
monotonicity and flow losses plus identity, freezes both discriminators, and backpropagates the
combined objective. The discriminator step re-enables discriminator parameters, samples from the
image history buffer, and updates `netD_A` and `netD_B` on real-vs-generated logits.

---

## 5. Implementation notes

- `models/__init__.py` resolves `--model` to `models/<name>_model.py` by class-name match;
  `--model axialflownet` selects `models/axialflownet_model.py`.
- `--texture_bound` and `--flow_mag_threshold` are registered by the model's
  `modify_commandline_options`. `flow_mag_threshold` is used only by the training-time magnitude
  penalty; reloading a checkpoint is unaffected by it.
- `flow_head.py` emits a `tanh`-bounded field in `[-1, 1]`, and `warp_with_flow` scales it by
  `(2/W, 2/H)`, so a flow value of `v` is exactly `v` pixels of displacement (§2.4).
- `scripts/inference.py` resolves a checkpoint path or directory and reads `inference_config.json`
  next to the weights for the training-time `texture_bound`, so a run directory needs no extra flags.
- Flow fields are colour-coded with the Middlebury optical-flow convention
  (`scripts/inference.py:flow_to_image`): direction to hue, magnitude to saturation, zero
  displacement to white.
