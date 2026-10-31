"""
AxialFlowNet Model — Geometric Flow-Field Image Translation.

Each generator (G_A, G_B) is split into two parallel branches:
  - Texture branch: G_resnet_main -> final_conv -> tanh. Predicts an RGB
    *texture correction* added to the warped input.
  - Flow branch:    G_resnet_main -> FlowHead -> tanh. Predicts a 2-channel
    flow field that warps the input image via grid_sample.

Final output:  fake = warp(real, flow) + texture

Initial behavior: FlowHead is zero-initialized, so flow = 0 at epoch 0 and
fake = warp(real, 0) + texture = real + texture.  The texture branch carries
the appearance change at first; the flow field gradually learns to predict
the axial-directional stretching as training progresses.

Training combines the adversarial and cycle-consistency objectives with an
axial-monotonicity penalty on the reconstruction error, an SSIM/Sobel
structural term on the cycle reconstruction, and three flow-specific
regularizers:
  - L_flow_smooth: total-variation on the flow field
  - L_flow_mono:   ReLU(-d(flow_y)/dy) — axial monotonicity of the flow
  - L_flow_mag:    soft penalty when |flow| > --flow_mag_threshold

Two architecture constants are exposed on the command line because they have
to match the weights being loaded:
  --texture_bound      (3-axial run: 0.3; paired mitochondrial run: 0.15)
  --flow_mag_threshold (3-axial run: 0.3; paired mitochondrial run: 0.4)
Both are recorded per run in `checkpoints/<run>/inference_config.json`.

Class name: AxialFlowNetModel  (so --model axialflownet resolves to it).
"""
import torch
import itertools
import functools
import torch.nn as nn
import torch.nn.functional as F
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .axial_d_networks import AxialMultiscaleDiscriminator
from .flow_head import FlowHead, warp_with_flow


class _GeometricGenerator(nn.Module):
    """G_resnet_main (truncated at the Tanh) + FlowHead + texture final layer.

    Produces (fake, flow) so the loss layer can supervise the flow field.
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=None,
                 use_dropout=False, n_blocks=9, padding_type="reflect",
                 texture_bound=0.15):
        super().__init__()
        self.texture_bound = float(texture_bound)
        if norm_layer is None:
            norm_layer = nn.InstanceNorm2d
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        assert n_blocks >= 0
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf), nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [_ResnetBlockShim(ngf * mult, padding_type, norm_layer,
                                       use_dropout, use_bias)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=3, stride=2, padding=1,
                                        output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        # Truncate before the final 7x7 conv + Tanh; we'll reattach.
        self.main = nn.Sequential(*model)

        # Texture branch: 7x7 reflection-pad conv + Tanh, predicts residual RGB
        self.texture_pad = nn.ReflectionPad2d(3)
        self.texture_conv = nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)

        # Flow branch: small FlowHead
        self.flow_head = FlowHead(in_channels=ngf, hidden=32)

    def forward(self, x):
        feat = self.main(x)
        # Texture
        t = self.texture_conv(self.texture_pad(feat))
        texture = torch.tanh(t) * self.texture_bound  # bounded residual (default ±0.15)
        # Flow
        flow = self.flow_head(feat)  # already bounded in [-1, 1]
        # Warp input with flow
        warped = warp_with_flow(x, flow)
        # Combine
        fake = warped + texture
        # Clamp to Tanh range for stability
        fake = torch.clamp(fake, -1.0, 1.0)
        return fake, flow


class _ResnetBlockShim(nn.Module):
    """A wrapper around ResnetBlock to keep the same interface as the original code."""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.block = networks.ResnetBlock(dim, padding_type, norm_layer,
                                          use_dropout, use_bias)

    def forward(self, x):
        return self.block(x)


class AxialFlowNetModel(BaseModel):
    """AxialFlowNet: geometric flow-field generator with dual-branch output."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument("--lambda_A", type=float, default=10.0)
            parser.add_argument("--lambda_B", type=float, default=10.0)
            parser.add_argument("--lambda_identity", type=float, default=0.5)
            parser.add_argument("--lambda_mono", type=float, default=2.0)
            parser.add_argument("--lambda_struct", type=float, default=1.0)
            parser.add_argument("--struct_ssim_w", type=float, default=1.0)
            parser.add_argument("--struct_edge_w", type=float, default=0.5)
            # New flow-related weights
            parser.add_argument("--lambda_flow_smooth", type=float, default=2.0,
                                help="weight of flow-field smoothness (TV) loss")
            parser.add_argument("--lambda_flow_mono", type=float, default=5.0,
                                help="weight of flow axial-monotonicity loss")
            parser.add_argument("--lambda_flow_mag", type=float, default=1.0,
                                help="weight of flow magnitude penalty when |flow| exceeds --flow_mag_threshold")
            parser.add_argument("--texture_bound", type=float, default=0.15,
                                help="maximum absolute value of the texture residual branch; "
                                     "must match the weights (see inference_config.json)")
            parser.add_argument("--flow_mag_threshold", type=float, default=0.4,
                                help="soft threshold above which the flow magnitude is penalised "
                                     "(training only; must match the weights)")
            parser.add_argument("--lr_D", type=float, default=None,
                                help="separate lr for D; default = opt.lr")
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["D_A", "G_A", "cycle_A", "cycle_A_struct", "cycle_A_mono",
                           "flow_smooth", "flow_mono", "flow_mag",
                           "idt_A", "D_B", "G_B", "cycle_B", "cycle_B_struct", "cycle_B_mono",
                           "idt_B"]
        visual_names_A = ["real_A", "fake_B", "rec_A"]
        visual_names_B = ["real_B", "fake_A", "rec_B"]
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append("idt_B")
            visual_names_B.append("idt_A")
        self.visual_names = visual_names_A + visual_names_B

        if self.isTrain:
            self.model_names = ["G_A", "G_B", "D_A", "D_B"]
        else:
            self.model_names = ["G_A", "G_B"]

        norm_layer = networks.get_norm_layer(norm_type=opt.norm)
        texture_bound = getattr(opt, "texture_bound", 0.15)
        self.netG_A = _GeometricGenerator(opt.input_nc, opt.output_nc, opt.ngf,
                                          norm_layer=norm_layer, use_dropout=False,
                                          n_blocks=9, texture_bound=texture_bound).to(self.device)
        self.netG_B = _GeometricGenerator(opt.output_nc, opt.input_nc, opt.ngf,
                                          norm_layer=norm_layer, use_dropout=False,
                                          n_blocks=9, texture_bound=texture_bound).to(self.device)

        if self.isTrain:
            self.netD_A = AxialMultiscaleDiscriminator(opt.output_nc, ndf=opt.ndf, norm=opt.norm).to(self.device)
            self.netD_B = AxialMultiscaleDiscriminator(opt.input_nc, ndf=opt.ndf, norm=opt.norm).to(self.device)

        if self.isTrain:
            if opt.lambda_identity > 0.0:
                assert opt.input_nc == opt.output_nc
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = nn.L1Loss()
            self.criterionIdt = nn.L1Loss()
            self.sobel_x = torch.tensor(
                [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32).repeat(3, 1, 1, 1).to(self.device)
            self.sobel_y = torch.tensor(
                [[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32).repeat(3, 1, 1, 1).to(self.device)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=(opt.lr_D if opt.lr_D is not None else opt.lr), betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def _generate(self, netG, x):
        fake, flow = netG(x)
        return fake, flow

    def forward(self):
        self.fake_B, self.flow_AtoB = self._generate(self.netG_A, self.real_A)
        self.rec_A, _ = self._generate(self.netG_B, self.fake_B)
        self.fake_A, self.flow_BtoA = self._generate(self.netG_B, self.real_B)
        self.rec_B, _ = self._generate(self.netG_A, self.fake_A)

    def _ssim(self, x, y):
        x = (x + 1) * 0.5
        y = (y + 1) * 0.5
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)
        sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
        sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
        num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        return (num / den).mean()

    def _sobel_edge_map(self, img):
        img = (img + 1) * 0.5
        gx = F.conv2d(img, self.sobel_x, padding=1, groups=3)
        gy = F.conv2d(img, self.sobel_y, padding=1, groups=3)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
        return mag.mean(dim=1, keepdim=True)

    def _cycle_struct_loss(self, rec, real):
        l1 = self.criterionCycle(rec, real)
        ssim_term = 1.0 - self._ssim(rec, real)
        edge_rec = self._sobel_edge_map(rec)
        edge_real = self._sobel_edge_map(real)
        edge_term = F.l1_loss(edge_rec, edge_real)
        w = self.opt
        return l1 + w.struct_ssim_w * ssim_term + w.struct_edge_w * edge_term

    def _axis_monotonicity_loss(self, rec, real):
        diff = rec - real
        d_dy = diff[:, :, 1:, :] - diff[:, :, :-1, :]
        penalty = F.relu(-d_dy)
        return penalty.mean()

    def _flow_smooth_loss(self, flow):
        """Total-variation penalty on the flow field, encouraging smoothness."""
        dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
        dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
        return (dx.abs().mean() + dy.abs().mean()) * 0.5

    def _flow_mono_loss(self, flow):
        """Axial monotonicity: the y-component (delta_y) should not decrease along y."""
        flow_y = flow[:, 1:2, :, :]
        d_dy = flow_y[:, :, 1:, :] - flow_y[:, :, :-1, :]
        penalty = F.relu(-d_dy)
        return penalty.mean()

    def _flow_mag_loss(self, flow, threshold=None):
        """Soft penalty when |flow| exceeds threshold.

        The threshold is exposed as `--flow_mag_threshold` so the released
        weight sets can each be reloaded with their own value.
        """
        if threshold is None:
            threshold = getattr(self.opt, "flow_mag_threshold", 0.4)
        mag = torch.sqrt(flow[:, 0:1] ** 2 + flow[:, 1:2] ** 2 + 1e-12)
        return F.relu(mag - threshold).mean()

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_mono = self.opt.lambda_mono
        lambda_struct = self.opt.lambda_struct
        l_fs = self.opt.lambda_flow_smooth
        l_fm = self.opt.lambda_flow_mono
        l_fmag = self.opt.lambda_flow_mag

        if lambda_idt > 0:
            self.idt_A, _ = self._generate(self.netG_A, self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B, _ = self._generate(self.netG_B, self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        self.loss_cycle_A_struct = self._cycle_struct_loss(self.rec_A, self.real_A) * lambda_A * lambda_struct
        self.loss_cycle_B_struct = self._cycle_struct_loss(self.rec_B, self.real_B) * lambda_B * lambda_struct

        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        self.loss_cycle_A_mono = self._axis_monotonicity_loss(self.rec_A, self.real_A) * lambda_mono
        self.loss_cycle_B_mono = self._axis_monotonicity_loss(self.rec_B, self.real_B) * lambda_mono

        # Flow-specific losses (Innovation 1)
        self.loss_flow_smooth = (self._flow_smooth_loss(self.flow_AtoB)
                                 + self._flow_smooth_loss(self.flow_BtoA)) * 0.5 * l_fs
        self.loss_flow_mono = (self._flow_mono_loss(self.flow_AtoB)
                               + self._flow_mono_loss(self.flow_BtoA)) * 0.5 * l_fm
        self.loss_flow_mag = (self._flow_mag_loss(self.flow_AtoB)
                              + self._flow_mag_loss(self.flow_BtoA)) * 0.5 * l_fmag

        self.loss_G = (self.loss_G_A + self.loss_G_B
                       + self.loss_cycle_A_struct + self.loss_cycle_B_struct
                       + self.loss_cycle_A_mono + self.loss_cycle_B_mono
                       + self.loss_flow_smooth + self.loss_flow_mono + self.loss_flow_mag
                       + self.loss_idt_A + self.loss_idt_B)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()