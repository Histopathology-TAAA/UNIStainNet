"""
UNIStainNet: Pixel-Space UNI-Guided Virtual Staining Network.

Architecture:
    Generator: SPADE-UNet conditioned on UNI pathology features + stain/class embedding
    Discriminator: Multi-scale PatchGAN (512 + 256)
    Losses: LPIPS@128 + adversarial + DAB intensity + DAB contrast

References:
    - Park et al., "Semantic Image Synthesis with SPADE" (CVPR 2019)
    - Chen et al., "A general-purpose self-supervised model for pathology" (Nature Medicine 2024)
    - Isola et al., "Image-to-Image Translation with pix2pix" (CVPR 2017)
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
import pytorch_lightning as pl
import torchvision
import wandb

from src.models.discriminator import (
    PatchDiscriminator, MultiScaleDiscriminator, ProjectionDiscriminator,
    hinge_loss_d, hinge_loss_g, r1_gradient_penalty, feature_matching_loss,
)
from src.models.generator import SPADEUNetGenerator
from src.models.losses import VGGFeatureExtractor, gram_matrix, PatchNCELoss
from src.utils.dab import DABExtractor


# ======================================================================
# Training Module
# ======================================================================

class UNIStainNetTrainer(pl.LightningModule):
    """PyTorch Lightning training module for UNIStainNet.

    Handles GAN training with manual optimization, CFG dropout, EMA, and
    all loss computations.
    """

    def __init__(
        self,
        # Architecture
        num_classes=5,
        null_class=4,
        class_dim=64,
        uni_dim=1024,
        ndf=64,
        disc_n_layers=3,
        input_skip=False,
        # Optimizer
        gen_lr=1e-4,
        disc_lr=4e-4,
        warmup_steps=1000,
        # Loss weights
        lpips_weight=1.0,
        lpips_256_weight=0.5,
        lpips_512_weight=0.0,
        l1_fullres_weight=1.0,
        lpips_fullres_weight=1.0,
        adversarial_weight=1.0,
        dab_intensity_weight=0.1,
        dab_contrast_weight=0.05,
        dab_sharpness_weight=0.0,
        gram_style_weight=0.0,
        edge_weight=0.0,
        he_edge_weight=0.0,
        bg_white_weight=0.0,
        bg_threshold=0.85,
        l1_lowres_weight=0.0,
        edge_encoder=False,
        edge_base_ch=32,
        uni_spatial_size=4,
        uncond_disc_weight=0.0,
        crop_disc_weight=0.0,
        crop_size=128,
        feat_match_weight=0.0,
        patchnce_weight=0.0,
        patchnce_layers=(2, 3, 4),
        patchnce_n_patches=256,
        patchnce_temperature=0.07,
        # Ablation
        disable_uni=False,
        disable_class=False,
        # GAN training
        r1_weight=10.0,
        r1_every=16,
        adversarial_start_step=2000,
        # CFG
        cfg_drop_class_prob=0.10,
        cfg_drop_uni_prob=0.10,
        cfg_drop_both_prob=0.05,
        # EMA
        ema_decay=0.999,
        # On-the-fly UNI extraction (for crop-based training)
        extract_uni_on_the_fly=False,
        uni_spatial_pool_size=32,
        # Resolution
        image_size=512,
        # 1024 architecture: extend UNI SPADE to 512 level
        uni_spade_at_512=False,
        # Per-label names for multi-stain logging
        label_names=None,
        # Mixed-domain routing: probability of Case B (H&E H-map, misaligned).
        # 0.5 = classic 50/50. Reduce to 0.25 in V2 (after Eosin encoder is added)
        # so 75% of steps get full-res aligned supervision. Do NOT drop below 0.20
        # or the model will suffer inference shock (H&E H-map at test time becomes OOD).
        case_b_prob=0.5,
        # Eosin bottleneck injection (requires trainA-E / valA-E data dirs)
        use_eosin_encoder=False,
        eosin_out_ch=64,
        # IHC-to-IHC Sobel edge loss (Case A only — pixel-aligned boundary supervision)
        # Disable: set ihc_edge_weight=0.0
        ihc_edge_weight=0.0,
        # DAB distribution matching: Wasserstein-1 on OD histograms (both cases)
        # Disable: set dab_histo_weight=0.0
        dab_histo_weight=0.0,
        # Block-level DAB mean matching (Case A only — requires spatial alignment)
        # Disable: set dab_block_weight=0.0
        dab_block_weight=0.0,
        dab_block_size=32,
        # Stain-conditioned projection discriminator (Miyato & Koyama, 2018)
        # Disable: set proj_disc_weight=0.0 — discriminator not instantiated at all
        proj_disc_weight=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.null_class = null_class

        # On-the-fly UNI feature extraction (loaded lazily on first use)
        self._uni_model = None
        self._uni_extract_on_the_fly = extract_uni_on_the_fly

        # Generator
        self.generator = SPADEUNetGenerator(
            num_classes=num_classes,
            class_dim=class_dim,
            uni_dim=uni_dim,
            input_skip=input_skip,
            edge_encoder=edge_encoder,
            edge_base_ch=edge_base_ch,
            uni_spatial_size=uni_spatial_size,
            image_size=image_size,
            uni_spade_at_512=uni_spade_at_512,
            use_eosin_encoder=use_eosin_encoder,
            eosin_out_ch=eosin_out_ch,
        )

        # Discriminator (global multi-scale) — only instantiate if adversarial loss is active.
        # With adversarial_weight=0 it is never trained and wastes VRAM.
        if adversarial_weight > 0:
            self.discriminator = MultiScaleDiscriminator(
                in_channels=3, ndf=ndf, n_layers=disc_n_layers,
            )
        else:
            self.discriminator = None

        # Crop discriminator (local full-res detail)
        if crop_disc_weight > 0:
            self.crop_discriminator = PatchDiscriminator(
                in_channels=3, ndf=ndf, n_layers=disc_n_layers,
            )
        else:
            self.crop_discriminator = None

        # Unconditional discriminator (alignment-free texture judge)
        # Also needed for feature matching loss (FM uses uncond disc features)
        if uncond_disc_weight > 0 or feat_match_weight > 0:
            self.uncond_discriminator = PatchDiscriminator(
                in_channels=3, ndf=ndf, n_layers=disc_n_layers,
            )
        else:
            self.uncond_discriminator = None

        # Stain-conditioned projection discriminator
        # Sees (IHC image, stain_label) → learns stain-specific realism criteria.
        # HER2: penalizes nuclear DAB; Ki67/ER/PR: rewards nuclear DAB.
        if proj_disc_weight > 0:
            self.proj_discriminator = ProjectionDiscriminator(
                in_channels=3, ndf=ndf, n_layers=disc_n_layers,
                num_stains=num_classes, stain_dim=class_dim,
            )
        else:
            self.proj_discriminator = None

        # PatchNCE loss (contrastive, alignment-free: H&E input vs generated)
        if patchnce_weight > 0:
            # Encoder channel dims: {1: 64, 2: 128, 3: 256, 4: 512}
            enc_channels = {1: 64, 2: 128, 3: 256, 4: 512}
            layer_channels = {l: enc_channels[l] for l in patchnce_layers}
            self.patchnce_loss = PatchNCELoss(
                layer_channels=layer_channels,
                num_patches=patchnce_n_patches,
                temperature=patchnce_temperature,
            )
        else:
            self.patchnce_loss = None

        # EMA generator
        self.generator_ema = copy.deepcopy(self.generator)
        self.generator_ema.requires_grad_(False)

        # Losses
        self.lpips_fn = lpips.LPIPS(net='alex')
        self.lpips_fn.requires_grad_(False)
        self.lpips_fn.eval()

        self.dab_extractor = DABExtractor(device='cpu')

        # VGG feature extractor for Gram-matrix style loss
        if gram_style_weight > 0:
            self.vgg_extractor = VGGFeatureExtractor()
            self.vgg_extractor.requires_grad_(False)
            self.vgg_extractor.eval()
        else:
            self.vgg_extractor = None

        # Param counts
        n_gen = sum(p.numel() for p in self.generator.parameters())
        n_disc = sum(p.numel() for p in self.discriminator.parameters()) if self.discriminator else 0
        n_crop = sum(p.numel() for p in self.crop_discriminator.parameters()) if self.crop_discriminator else 0
        n_uncond = sum(p.numel() for p in self.uncond_discriminator.parameters()) if self.uncond_discriminator else 0
        n_proj = sum(p.numel() for p in self.proj_discriminator.parameters()) if self.proj_discriminator else 0
        print(f"Generator: {n_gen:,} params")
        print(f"Discriminator: {n_disc:,} (global) + {n_crop:,} (crop) + {n_uncond:,} (uncond) + {n_proj:,} (proj-stain)")

    def configure_optimizers(self):
        gen_params = list(self.generator.parameters())
        if self.patchnce_loss is not None:
            gen_params += list(self.patchnce_loss.parameters())
        opt_g = torch.optim.Adam(
            gen_params,
            lr=self.hparams.gen_lr,
            betas=(0.0, 0.999),
        )
        # All discriminator params in one optimizer
        disc_params = list(self.discriminator.parameters()) if self.discriminator else []
        if self.crop_discriminator is not None:
            disc_params += list(self.crop_discriminator.parameters())
        if self.uncond_discriminator is not None:
            disc_params += list(self.uncond_discriminator.parameters())
        if self.proj_discriminator is not None:
            disc_params += list(self.proj_discriminator.parameters())
        # PyTorch Adam requires a non-empty param list. When all discriminators are
        # disabled (e.g. sanity-check / ablation), register a dummy zero-grad param
        # so Lightning's two-optimizer contract is still satisfied.
        if not disc_params:
            self._disc_dummy = nn.Parameter(torch.zeros(1), requires_grad=True)
            disc_params = [self._disc_dummy]
        opt_d = torch.optim.Adam(
            disc_params,
            lr=self.hparams.disc_lr,
            betas=(0.0, 0.999),
        )
        return [opt_g, opt_d]

    def _get_lr_scale(self):
        """Linear warmup."""
        if self.global_step < self.hparams.warmup_steps:
            return self.global_step / max(1, self.hparams.warmup_steps)
        return 1.0

    @torch.no_grad()
    def _update_ema(self):
        """Update EMA generator weights."""
        decay = self.hparams.ema_decay
        for p_ema, p in zip(self.generator_ema.parameters(), self.generator.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)

    def on_save_checkpoint(self, checkpoint):
        """Exclude frozen UNI model from checkpoint (it's reloaded on-the-fly)."""
        state_dict = checkpoint.get('state_dict', {})
        keys_to_remove = [k for k in state_dict if k.startswith('_uni_model.')]
        for k in keys_to_remove:
            del state_dict[k]

    def on_load_checkpoint(self, checkpoint):
        """Filter out UNI model keys from old checkpoints that included them."""
        state_dict = checkpoint.get('state_dict', {})
        keys_to_remove = [k for k in state_dict if k.startswith('_uni_model.')]
        for k in keys_to_remove:
            del state_dict[k]

    def _load_uni_model(self):
        """Lazily load UNI ViT-L/16 for on-the-fly feature extraction."""
        if self._uni_model is None:
            import timm
            self._uni_model = timm.create_model(
                "hf-hub:MahmoodLab/uni",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=True,
            )
            self._uni_model.eval()
            self._uni_model.requires_grad_(False)
            self._uni_model = self._uni_model.to(self.device)
            n_params = sum(p.numel() for p in self._uni_model.parameters())
            print(f"UNI model loaded for on-the-fly extraction: {n_params:,} params")
        return self._uni_model

    @torch.no_grad()
    def _extract_uni_from_sub_crops(self, uni_sub_crops):
        """Extract UNI features from pre-prepared sub-crops on GPU.

        Args:
            uni_sub_crops: [B, 16, 3, 224, 224] — batch of 4x4 sub-crop grids,
                           already normalized with ImageNet stats.

        Returns:
            uni_features: [B, S*S, 1024] where S = uni_spatial_pool_size (default 32)
        """
        uni_model = self._load_uni_model()
        B = uni_sub_crops.shape[0]
        spatial_size = self.hparams.uni_spatial_pool_size
        num_crops = 4  # 4x4 grid
        patches_per_side = 14  # 224/16

        # Batched UNI forward: [B, 16, 3, 224, 224] -> [B*16, 3, 224, 224]
        all_crops = uni_sub_crops.reshape(B * 16, 3, 224, 224).to(self.device)
        all_feats = uni_model.forward_features(all_crops)  # [B*16, 197, 1024]
        patch_tokens = all_feats[:, 1:, :]  # [B*16, 196, 1024]

        # Reshape back to per-sample grids: [B, 4, 4, 14, 14, 1024]
        patch_tokens = patch_tokens.reshape(
            B, num_crops, num_crops,
            patches_per_side, patches_per_side, 1024
        )
        # Interleave to spatial grid: [B, 56, 56, 1024]
        full_size = num_crops * patches_per_side  # 56
        full_grid = patch_tokens.permute(0, 1, 3, 2, 4, 5)
        full_grid = full_grid.reshape(B, full_size, full_size, 1024)

        # Pool to target spatial size (batched)
        if spatial_size < full_size:
            grid_bchw = full_grid.permute(0, 3, 1, 2)  # [B, 1024, 56, 56]
            pooled = F.adaptive_avg_pool2d(grid_bchw, spatial_size)  # [B, 1024, S, S]
            result = pooled.permute(0, 2, 3, 1)  # [B, S, S, 1024]
        else:
            result = full_grid

        S = result.shape[1]
        return result.reshape(B, S * S, 1024)  # [B, S*S, 1024]

    def _prepare_uni_sub_crops_from_tensor(self, he_rgb):
        """Prepare UNI sub-crops from a normalized H&E tensor.

        Args:
            he_rgb: [B, 3, H, W] in [-1, 1]

        Returns:
            [B, 16, 3, 224, 224] ImageNet-normalized sub-crops
        """
        B, C, H, W = he_rgb.shape
        grid = 4
        if H % grid != 0 or W % grid != 0:
            raise ValueError(f"H&E size {H}x{W} must be divisible by {grid}")

        he_01 = ((he_rgb + 1) / 2).clamp(0, 1)
        ph = H // grid
        pw = W // grid
        patches = he_01.reshape(B, C, grid, ph, grid, pw)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(B * grid * grid, C, ph, pw)

        patches = F.interpolate(patches, size=(224, 224), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=he_rgb.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=he_rgb.device).view(1, 3, 1, 1)
        patches = (patches - mean) / std

        return patches.reshape(B, grid * grid, 3, 224, 224)

    def _apply_cfg_dropout(self, labels, uni_features):
        """Apply classifier-free guidance dropout during training (vectorized)."""
        B = labels.shape[0]
        device = labels.device

        new_labels = labels.clone()
        new_uni = uni_features.clone()

        r = torch.rand(B, device=device)
        p_both = self.hparams.cfg_drop_both_prob
        p_class = p_both + self.hparams.cfg_drop_class_prob
        p_uni = p_class + self.hparams.cfg_drop_uni_prob

        drop_both = r < p_both
        drop_class = (r >= p_both) & (r < p_class)
        drop_uni = (r >= p_class) & (r < p_uni)

        new_labels[drop_both | drop_class] = self.null_class
        new_uni[drop_both | drop_uni] = 0.0

        return new_labels, new_uni

    def compute_dab_intensity_loss(self, generated, target):
        """Top-10% percentile matching for DAB intensity."""
        with torch.amp.autocast('cuda', enabled=False):
            gen = generated.float()
            tgt = target.float()

            dab_gen = self.dab_extractor.extract_dab_intensity(gen, normalize="none")
            dab_tgt = self.dab_extractor.extract_dab_intensity(tgt, normalize="none")

            def _batched_top10_mean(dab):
                """Compute mean of top-10% DAB intensity per sample (vectorized)."""
                B = dab.shape[0]
                flat = dab.reshape(B, -1)  # [B, H*W]
                p99 = torch.quantile(flat, 0.99, dim=1, keepdim=True)
                flat = flat.clamp(max=p99)
                p90 = torch.quantile(flat, 0.9, dim=1, keepdim=True)
                mask = flat >= p90  # [B, H*W]
                # Use masked mean: sum(vals * mask) / sum(mask), fallback to flat mean
                masked_sum = (flat * mask).sum(dim=1)
                mask_count = mask.sum(dim=1).clamp(min=1)
                return masked_sum / mask_count  # [B]

            gen_scores = _batched_top10_mean(dab_gen)
            tgt_scores = _batched_top10_mean(dab_tgt)
            return F.l1_loss(gen_scores, tgt_scores)

    def compute_dab_contrast_loss(self, generated, labels):
        """Class-ordering hinge loss: DAB(3+) > DAB(2+) > DAB(1+) > DAB(0)."""
        with torch.amp.autocast('cuda', enabled=False):
            gen = generated.float()
            # Only use non-null labels
            valid = labels < self.null_class
            if valid.sum() < 2:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            gen_valid = gen[valid]
            labels_valid = labels[valid]

            dab_gen = self.dab_extractor.extract_dab_intensity(gen_valid, normalize="none")

            B = dab_gen.shape[0]
            flat = dab_gen.reshape(B, -1)
            p99 = torch.quantile(flat, 0.99, dim=1, keepdim=True)
            flat = flat.clamp(max=p99)
            p90 = torch.quantile(flat, 0.9, dim=1, keepdim=True)
            mask = flat >= p90
            masked_sum = (flat * mask).sum(dim=1)
            mask_count = mask.sum(dim=1).clamp(min=1)
            dab_scores = masked_sum / mask_count

            class_pairs = [
                (3, 0, 0.20), (3, 1, 0.15),
                (2, 0, 0.08), (3, 2, 0.10),
            ]

            losses = []
            for high_cls, low_cls, margin in class_pairs:
                high_mask = labels_valid == high_cls
                low_mask = labels_valid == low_cls
                if high_mask.sum() > 0 and low_mask.sum() > 0:
                    high_score = dab_scores[high_mask].mean()
                    low_score = dab_scores[low_mask].mean()
                    losses.append(F.relu(margin - (high_score - low_score)))

            if losses:
                return torch.stack(losses).mean()
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def compute_edge_loss(self, generated, target):
        """Fourier spectral loss at 256x256 for boundary sharpness.

        Compares power spectrum magnitudes between generated and target.
        The Fourier magnitude is inherently translation-invariant — shifting
        an image doesn't change its frequency content — so this is robust to
        the ~30px misalignment in consecutive-cut BCI pairs.

        Focuses on high-frequency bands (outer 75% of spectrum) where
        blurriness manifests as reduced power.
        """
        with torch.amp.autocast('cuda', enabled=False):
            gen = F.interpolate(generated.float(), size=256, mode='bilinear', align_corners=False)
            tgt = F.interpolate(target.float(), size=256, mode='bilinear', align_corners=False)

            # Grayscale
            gen_gray = gen.mean(dim=1, keepdim=True)
            tgt_gray = tgt.mean(dim=1, keepdim=True)

            # 2D FFT -> power spectrum (log-scale for stability)
            gen_fft = torch.fft.fft2(gen_gray)
            tgt_fft = torch.fft.fft2(tgt_gray)
            gen_mag = torch.log1p(gen_fft.abs())
            tgt_mag = torch.log1p(tgt_fft.abs())

            # High-frequency mask: keep outer 75% of spectrum
            H, W = gen_mag.shape[-2], gen_mag.shape[-1]
            cy, cx = H // 2, W // 2
            y = torch.arange(H, device=gen.device).float() - cy
            x = torch.arange(W, device=gen.device).float() - cx
            dist = (y[:, None] ** 2 + x[None, :] ** 2).sqrt()
            max_dist = (cy ** 2 + cx ** 2) ** 0.5
            hf_mask = (dist > 0.25 * max_dist).float()

            # L1 on high-frequency magnitudes
            return F.l1_loss(gen_mag * hf_mask, tgt_mag * hf_mask)

    def compute_dab_sharpness_loss(self, generated, target):
        """DAB spatial sharpness loss: penalizes diffuse brown, rewards membrane-localized DAB.

        Two components:
        1. DAB gradient magnitude: mean Sobel gradient magnitude per image.
        2. DAB local variance distribution: sorted-L1 (Wasserstein-1) on
           patch variance vectors.
        """
        with torch.amp.autocast('cuda', enabled=False):
            gen = generated.float()
            tgt = target.float()

            dab_gen = self.dab_extractor.extract_dab_intensity(gen, normalize="none")
            dab_tgt = self.dab_extractor.extract_dab_intensity(tgt, normalize="none")

            # Ensure [B, 1, H, W]
            if dab_gen.dim() == 3:
                dab_gen = dab_gen.unsqueeze(1)
            if dab_tgt.dim() == 3:
                dab_tgt = dab_tgt.unsqueeze(1)

            B = dab_gen.shape[0]

            # --- Component 1: Gradient magnitude (batched) ---
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=torch.float32, device=gen.device).view(1, 1, 3, 3)
            sobel_y = sobel_x.transpose(-1, -2)

            gx_gen = F.conv2d(dab_gen, sobel_x, padding=1)
            gy_gen = F.conv2d(dab_gen, sobel_y, padding=1)
            grad_gen = (gx_gen**2 + gy_gen**2 + 1e-8).sqrt()

            gx_tgt = F.conv2d(dab_tgt, sobel_x, padding=1)
            gy_tgt = F.conv2d(dab_tgt, sobel_y, padding=1)
            grad_tgt = (gx_tgt**2 + gy_tgt**2 + 1e-8).sqrt()

            # Match mean gradient magnitude per image
            grad_loss = F.l1_loss(grad_gen.mean(dim=[1, 2, 3]), grad_tgt.mean(dim=[1, 2, 3]))

            # --- Component 2: Local variance distribution (sorted-L1) ---
            ps = 16  # patch size
            var_losses = []
            for i in range(B):
                g = dab_gen[i, 0]  # [H, W]
                t = dab_tgt[i, 0]

                H, W = g.shape
                nH, nW = H // ps, W // ps
                g_patches = g[:nH*ps, :nW*ps].reshape(nH, ps, nW, ps).permute(0, 2, 1, 3).reshape(-1, ps*ps)
                t_patches = t[:nH*ps, :nW*ps].reshape(nH, ps, nW, ps).permute(0, 2, 1, 3).reshape(-1, ps*ps)

                g_var = g_patches.var(dim=1)
                t_var = t_patches.var(dim=1)

                g_sorted, _ = g_var.sort()
                t_sorted, _ = t_var.sort()
                var_losses.append(F.l1_loss(g_sorted, t_sorted.detach()))

            var_loss = torch.stack(var_losses).mean()

            return grad_loss + var_loss

    def compute_he_edge_loss(self, generated, he_input):
        """H&E edge structure preservation loss.

        Extracts Sobel edges from H&E input and generated output, then
        computes L1 loss between edge maps at multiple scales.
        """
        with torch.amp.autocast('cuda', enabled=False):
            gen = generated.float()
            he = he_input.float()

            gen_gray = ((gen + 1) / 2).mean(dim=1, keepdim=True)
            he_gray = ((he + 1) / 2).mean(dim=1, keepdim=True)

            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=torch.float32, device=gen.device).view(1, 1, 3, 3)
            sobel_y = sobel_x.transpose(-1, -2)

            loss = 0.0
            full_size = gen_gray.shape[-1]
            scales = [full_size, full_size // 2]
            for size in scales:
                if size < full_size:
                    g = F.interpolate(gen_gray, size=size, mode='bilinear', align_corners=False)
                    h = F.interpolate(he_gray, size=size, mode='bilinear', align_corners=False)
                else:
                    g, h = gen_gray, he_gray

                gx_gen = F.conv2d(g, sobel_x, padding=1)
                gy_gen = F.conv2d(g, sobel_y, padding=1)
                edge_gen = (gx_gen**2 + gy_gen**2 + 1e-8).sqrt()

                gx_he = F.conv2d(h, sobel_x, padding=1)
                gy_he = F.conv2d(h, sobel_y, padding=1)
                edge_he = (gx_he**2 + gy_he**2 + 1e-8).sqrt()

                loss = loss + F.l1_loss(edge_gen, edge_he.detach())

            return loss / 2.0

    def compute_background_loss(self, generated, he_input):
        """Background white loss: push background regions toward white."""
        with torch.amp.autocast('cuda', enabled=False):
            gen = generated.float()
            he = he_input.float()

            he_bright = ((he + 1) / 2).mean(dim=1, keepdim=True)

            threshold = self.hparams.bg_threshold
            mask = torch.sigmoid((he_bright - threshold) * 20.0)

            white_target = torch.ones_like(gen)
            diff = (gen - white_target).abs()

            weighted_diff = diff * mask

            mask_sum = mask.sum() * 3
            if mask_sum > 0:
                return weighted_diff.sum() / mask_sum
            return torch.tensor(0.0, device=gen.device, requires_grad=True)

    def compute_gram_style_loss(self, generated, target):
        """Gram-matrix style loss: match texture statistics via VGG feature correlations."""
        with torch.amp.autocast('cuda', enabled=False):
            gen = generated.float()
            tgt = target.float()

            gen_256 = F.interpolate(gen, size=256, mode='bilinear', align_corners=False)
            tgt_256 = F.interpolate(tgt, size=256, mode='bilinear', align_corners=False)

            gen_feats = self.vgg_extractor(gen_256)
            tgt_feats = self.vgg_extractor(tgt_256)

            loss = 0.0
            for gf, tf in zip(gen_feats, tgt_feats):
                gram_gen = gram_matrix(gf)
                gram_tgt = gram_matrix(tf)
                loss = loss + F.l1_loss(gram_gen, gram_tgt.detach())

            return loss / len(gen_feats)

    def compute_dab_histogram_loss(self, generated, target):
        """Wasserstein-1 on DAB optical density distributions.

        Compares the sorted OD value distributions (not spatial layout) so it
        is alignment-free — valid in both Case A (aligned) and Case B (misaligned).
        Catches distribution shape errors: e.g., a uniform 1+ output that fools
        the mean-intensity loss but has the wrong bimodal 0/3+ distribution.

        Disable: set dab_histo_weight=0.0
        """
        with torch.amp.autocast('cuda', enabled=False):
            gen = generated.float()
            tgt = target.float()

            dab_gen = self.dab_extractor.extract_dab_intensity(gen, normalize="none")
            dab_tgt = self.dab_extractor.extract_dab_intensity(tgt, normalize="none")

            B = dab_gen.shape[0]
            gen_flat = dab_gen.reshape(B, -1)   # [B, H*W]
            tgt_flat = dab_tgt.reshape(B, -1)

            # Clamp at 99th percentile of target to suppress outlier noise
            p99 = torch.quantile(tgt_flat, 0.99, dim=1, keepdim=True)
            gen_flat = gen_flat.clamp(max=p99)
            tgt_flat = tgt_flat.clamp(max=p99)

            # Sorted L1 = Wasserstein-1 between OD distributions
            gen_sorted, _ = gen_flat.sort(dim=1)
            tgt_sorted, _ = tgt_flat.sort(dim=1)

            return F.l1_loss(gen_sorted, tgt_sorted.detach())

    def compute_dab_block_loss(self, generated, target):
        """Per-block DAB mean matching — spatial DAB distribution supervisor.

        Divides the image into dab_block_size×dab_block_size patches and matches
        per-block DAB means. Forces the generator to place DAB *where* the ground
        truth has it, not just match the global intensity.

        CASE A ONLY — requires pixel-aligned IHC ground truth. Using in Case B
        would be equivalent to the feature-matching spatial-alignment bug.

        Disable: set dab_block_weight=0.0
        """
        with torch.amp.autocast('cuda', enabled=False):
            gen = generated.float()
            tgt = target.float()

            dab_gen = self.dab_extractor.extract_dab_intensity(gen, normalize="none")
            dab_tgt = self.dab_extractor.extract_dab_intensity(tgt, normalize="none")

            if dab_gen.dim() == 3:
                dab_gen = dab_gen.unsqueeze(1)   # [B, 1, H, W]
            if dab_tgt.dim() == 3:
                dab_tgt = dab_tgt.unsqueeze(1)

            bs = self.hparams.dab_block_size
            gen_blocks = F.avg_pool2d(dab_gen, kernel_size=bs, stride=bs)
            tgt_blocks = F.avg_pool2d(dab_tgt, kernel_size=bs, stride=bs)

            return F.l1_loss(gen_blocks, tgt_blocks.detach())

    def training_step(self, batch, batch_idx):
        he_rgb, ihc_rgb, he_h_map, ihc_h_map, he_e_map, labels, fnames = batch
        opt_g, opt_d = self.optimizers()

        # On-the-fly UNI extraction from H&E RGB
        if self._uni_extract_on_the_fly:
            uni_sub_crops = self._prepare_uni_sub_crops_from_tensor(he_rgb)
            uni = self._extract_uni_from_sub_crops(uni_sub_crops)
        else:
            raise ValueError("UNI features must be provided or extracted on-the-fly.")

        # Apply CFG dropout
        labels_dropped, uni_dropped = self._apply_cfg_dropout(labels, uni)

        # Ablation: zero out UNI features
        if self.hparams.disable_uni:
            uni_dropped = torch.zeros_like(uni_dropped)

        # Ablation: force all labels to null class
        if self.hparams.disable_class:
            labels_dropped = torch.full_like(labels_dropped, self.hparams.null_class)

        # ----------------------------------------------------------------
        # Generator step
        # ----------------------------------------------------------------
        use_aligned = torch.rand(1, device=self.device) > self.hparams.case_b_prob
        gen_input = ihc_h_map if use_aligned else he_h_map
        generated = self.generator(gen_input, uni_dropped, labels_dropped, e_maps=he_e_map)

        loss_g = torch.tensor(0.0, device=self.device)
        loss_lpips = torch.tensor(0.0, device=self.device)

        if use_aligned:
            # Full-res L1 and LPIPS against IHC target
            loss_l1_full = F.l1_loss(generated, ihc_rgb)
            loss_lpips_full = self.lpips_fn(generated, ihc_rgb).mean()
            loss_g = loss_g + self.hparams.l1_fullres_weight * loss_l1_full
            loss_g = loss_g + self.hparams.lpips_fullres_weight * loss_lpips_full
            loss_lpips = loss_lpips_full
            self.log('train/l1_fullres', loss_l1_full, prog_bar=False)
            self.log('train/lpips_fullres', loss_lpips_full, prog_bar=False)
        else:
            # Misalignment-aware losses.
            # l1_lowres (64×64) is replaced by FFT spectral loss which is
            # translation-invariant by construction: shifting an image does not
            # change its frequency spectrum. This gives a genuine structural signal
            # (sharpness, texture grain, stain density frequency) without penalising
            # the 30px spatial shift between consecutive tissue sections.
            if self.hparams.l1_lowres_weight > 0:
                loss_spectral = self.compute_edge_loss(generated, ihc_rgb)
                loss_g = loss_g + self.hparams.l1_lowres_weight * loss_spectral
                self.log('train/spectral_misalign', loss_spectral, prog_bar=False)

            lpips_main_size = self.hparams.image_size // 4
            gen_lpips = F.interpolate(generated, size=lpips_main_size, mode='bilinear', align_corners=False)
            tgt_lpips = F.interpolate(ihc_rgb, size=lpips_main_size, mode='bilinear', align_corners=False)
            loss_lpips = self.lpips_fn(gen_lpips, tgt_lpips).mean()
            loss_g = loss_g + self.hparams.lpips_weight * loss_lpips

            if self.hparams.lpips_256_weight > 0:
                lpips_fine_size = self.hparams.image_size // 2
                gen_fine = F.interpolate(generated, size=lpips_fine_size, mode='bilinear', align_corners=False)
                tgt_fine = F.interpolate(ihc_rgb, size=lpips_fine_size, mode='bilinear', align_corners=False)
                loss_lpips_256 = self.lpips_fn(gen_fine, tgt_fine).mean()
                loss_g = loss_g + self.hparams.lpips_256_weight * loss_lpips_256
                self.log('train/lpips_fine', loss_lpips_256, prog_bar=False)

        # DAB losses (use original labels, not dropped)
        if self.hparams.dab_intensity_weight > 0:
            loss_dab = self.compute_dab_intensity_loss(generated, ihc_rgb)
            loss_g = loss_g + self.hparams.dab_intensity_weight * loss_dab
            self.log('train/dab_intensity', loss_dab, prog_bar=False)

        # DAB histogram (Wasserstein-1): distribution shape, both cases
        if self.hparams.dab_histo_weight > 0:
            loss_dab_histo = self.compute_dab_histogram_loss(generated, ihc_rgb)
            loss_g = loss_g + self.hparams.dab_histo_weight * loss_dab_histo
            self.log('train/dab_histo', loss_dab_histo, prog_bar=False)

        # DAB block: spatial DAB placement, Case A only
        if self.hparams.dab_block_weight > 0 and use_aligned:
            loss_dab_block = self.compute_dab_block_loss(generated, ihc_rgb)
            loss_g = loss_g + self.hparams.dab_block_weight * loss_dab_block
            self.log('train/dab_block', loss_dab_block, prog_bar=False)

        if self.hparams.dab_contrast_weight > 0:
            # Use labels_dropped: samples where class was CFG-dropped to null_class
            # are excluded inside compute_dab_contrast_loss (valid = labels < null_class).
            loss_dab_contrast = self.compute_dab_contrast_loss(generated, labels_dropped)
            loss_g = loss_g + self.hparams.dab_contrast_weight * loss_dab_contrast
            self.log('train/dab_contrast', loss_dab_contrast, prog_bar=False)

        # Edge loss (boundary sharpness)
        if self.hparams.edge_weight > 0:
            loss_edge = self.compute_edge_loss(generated, ihc_rgb)
            loss_g = loss_g + self.hparams.edge_weight * loss_edge
            self.log('train/edge_loss', loss_edge, prog_bar=False)

        # DAB sharpness loss (membrane-localized vs diffuse brown)
        if self.hparams.dab_sharpness_weight > 0:
            loss_dab_sharp = self.compute_dab_sharpness_loss(generated, ihc_rgb)
            loss_g = loss_g + self.hparams.dab_sharpness_weight * loss_dab_sharp
            self.log('train/dab_sharpness', loss_dab_sharp, prog_bar=False)

        # Gram-matrix style loss
        if self.hparams.gram_style_weight > 0 and self.vgg_extractor is not None:
            loss_gram = self.compute_gram_style_loss(generated, ihc_rgb)
            loss_g = loss_g + self.hparams.gram_style_weight * loss_gram
            self.log('train/gram_style', loss_gram, prog_bar=False)

        # IHC-to-IHC Sobel edge loss: Case A only, pixel-aligned.
        # Forces generator to reproduce IHC membrane/boundary structure,
        # not just pixel color. Complements LPIPS which is more holistic.
        if self.hparams.ihc_edge_weight > 0 and use_aligned:
            loss_ihc_edge = self.compute_he_edge_loss(generated, ihc_rgb)
            loss_g = loss_g + self.hparams.ihc_edge_weight * loss_ihc_edge
            self.log('train/ihc_edge', loss_ihc_edge, prog_bar=False)

        # H&E edge structure preservation — only in Case B (H&E H-map input).
        # In Case A the generator input is the IHC H-map, so comparing generated
        # IHC edges to H&E edges is comparing different tissue sections, which
        # introduces conflicting gradients against the full-res IHC supervision.
        if self.hparams.he_edge_weight > 0 and not use_aligned:
            loss_he_edge = self.compute_he_edge_loss(generated, he_rgb)
            loss_g = loss_g + self.hparams.he_edge_weight * loss_he_edge
            self.log('train/he_edge', loss_he_edge, prog_bar=False)

        # Background white loss
        if self.hparams.bg_white_weight > 0:
            loss_bg = self.compute_background_loss(generated, he_rgb)
            loss_g = loss_g + self.hparams.bg_white_weight * loss_bg
            self.log('train/bg_white', loss_bg, prog_bar=False)

        # PatchNCE loss (contrastive: H&E input vs generated, never sees GT)
        if self.hparams.patchnce_weight > 0 and self.patchnce_loss is not None:
            feats_he = self.generator.encode(he_h_map)
            feats_gen = self.generator.encode(generated)
            loss_nce = self.patchnce_loss(feats_he, feats_gen)
            loss_g = loss_g + self.hparams.patchnce_weight * loss_nce
            self.log('train/patchnce', loss_nce, prog_bar=False)

        # Adversarial losses (after warmup)
        loss_adv = torch.tensor(0.0, device=self.device)
        loss_feat_match = torch.tensor(0.0, device=self.device)
        loss_crop_adv = torch.tensor(0.0, device=self.device)
        loss_uncond_adv = torch.tensor(0.0, device=self.device)
        loss_proj_adv = torch.tensor(0.0, device=self.device)
        any_adv = (self.hparams.adversarial_weight > 0 or
                   self.hparams.uncond_disc_weight > 0 or
                   self.hparams.crop_disc_weight > 0 or
                   self.hparams.feat_match_weight > 0 or
                   self.hparams.proj_disc_weight > 0)
        img_sz = self.hparams.image_size
        # Pre-compute disc-resolution tensors (512 for 1024 input, identity for 512)
        if img_sz == 1024:
            ihc_for_disc = F.interpolate(ihc_rgb, size=512, mode='bilinear', align_corners=False)
        else:
            ihc_for_disc = ihc_rgb
        if self.global_step >= self.hparams.adversarial_start_step and any_adv:
            if img_sz == 1024:
                gen_for_disc = F.interpolate(generated, size=512, mode='bilinear', align_corners=False)
            else:
                gen_for_disc = generated

            if self.hparams.adversarial_weight > 0:
                disc_outputs = self.discriminator(gen_for_disc)
                loss_adv = sum(hinge_loss_g(out) for out in disc_outputs) / len(disc_outputs)
                loss_g = loss_g + self.hparams.adversarial_weight * loss_adv

            # Feature matching from unconditional disc.
            # Restricted to Case A (aligned) only. In Case B the discriminator's
            # spatial features reflect the ~30px slice misalignment — the gradient
            # tells the generator "your nuclei are in the wrong place," which it
            # cannot fix. The model responds by diffusing the stain to cover both
            # positions, producing the "light brown wash" / over-expressiveness.
            if (self.hparams.feat_match_weight > 0 and
                    self.uncond_discriminator is not None and use_aligned):
                _, fake_feats = self.uncond_discriminator(gen_for_disc, return_features=True)
                with torch.no_grad():
                    _, real_feats = self.uncond_discriminator(ihc_for_disc, return_features=True)
                loss_feat_match = feature_matching_loss(fake_feats, real_feats)
                loss_g = loss_g + self.hparams.feat_match_weight * loss_feat_match

            # Crop discriminator: random crops at full resolution
            if self.crop_discriminator is not None and self.hparams.crop_disc_weight > 0:
                cs = self.hparams.crop_size
                top = torch.randint(0, img_sz - cs, (1,)).item()
                left = torch.randint(0, img_sz - cs, (1,)).item()
                fake_crop = generated[:, :, top:top+cs, left:left+cs]
                loss_crop_adv = hinge_loss_g(self.crop_discriminator(fake_crop))
                loss_g = loss_g + self.hparams.crop_disc_weight * loss_crop_adv

            # Unconditional discriminator: alignment-free adversarial
            if self.uncond_discriminator is not None and self.hparams.uncond_disc_weight > 0:
                loss_uncond_adv = hinge_loss_g(self.uncond_discriminator(gen_for_disc))
                loss_g = loss_g + self.hparams.uncond_disc_weight * loss_uncond_adv

            # Stain-conditioned projection discriminator.
            # Uses labels_dropped so CFG-dropped samples (null_class) see the null embedding,
            # which the discriminator associates with generic IHC realism.
            # True stain labels (HER2/Ki67/ER/PR) activate stain-specific realism criteria.
            if self.proj_discriminator is not None and self.hparams.proj_disc_weight > 0:
                proj_out = self.proj_discriminator(gen_for_disc, labels_dropped)
                loss_proj_adv = hinge_loss_g(proj_out)
                loss_g = loss_g + self.hparams.proj_disc_weight * loss_proj_adv

        # Generator backward + step
        lr_scale = self._get_lr_scale()
        for pg in opt_g.param_groups:
            pg['lr'] = self.hparams.gen_lr * lr_scale

        opt_g.zero_grad()
        self.manual_backward(loss_g)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        opt_g.step()

        # Update EMA
        self._update_ema()

        # ----------------------------------------------------------------
        # Discriminator step
        # ----------------------------------------------------------------
        loss_d = torch.tensor(0.0, device=self.device)
        loss_crop_d = torch.tensor(0.0, device=self.device)
        loss_uncond_d = torch.tensor(0.0, device=self.device)
        loss_proj_d = torch.tensor(0.0, device=self.device)
        if self.global_step >= self.hparams.adversarial_start_step and any_adv:
            with torch.no_grad():
                fake_detached = self.generator(gen_input, uni_dropped, labels_dropped, e_maps=he_e_map)

            # For 1024, downsample for disc
            if img_sz == 1024:
                fake_det_disc = F.interpolate(fake_detached, size=512, mode='bilinear', align_corners=False)
            else:
                fake_det_disc = fake_detached

            if self.hparams.adversarial_weight > 0:
                disc_real = self.discriminator(ihc_for_disc)
                disc_fake = self.discriminator(fake_det_disc)

                loss_d = sum(
                    hinge_loss_d(dr, df)
                    for dr, df in zip(disc_real, disc_fake)
                ) / len(disc_real)

            # Crop discriminator
            if self.crop_discriminator is not None and self.hparams.crop_disc_weight > 0:
                cs = self.hparams.crop_size
                top = torch.randint(0, img_sz - cs, (1,)).item()
                left = torch.randint(0, img_sz - cs, (1,)).item()
                real_crop = ihc_rgb[:, :, top:top+cs, left:left+cs]
                fake_crop = fake_detached[:, :, top:top+cs, left:left+cs]
                loss_crop_d = hinge_loss_d(
                    self.crop_discriminator(real_crop),
                    self.crop_discriminator(fake_crop),
                )
                loss_d = loss_d + self.hparams.crop_disc_weight * loss_crop_d

            # Unconditional discriminator
            if self.uncond_discriminator is not None and (
                    self.hparams.uncond_disc_weight > 0 or self.hparams.feat_match_weight > 0):
                uncond_real_out = self.uncond_discriminator(ihc_for_disc)
                uncond_fake_out = self.uncond_discriminator(fake_det_disc)
                loss_uncond_d = hinge_loss_d(uncond_real_out, uncond_fake_out)
                loss_d = loss_d + max(self.hparams.uncond_disc_weight, 1.0) * loss_uncond_d

            # Stain-conditioned projection discriminator.
            # Real images: (ihc_rgb, true_labels) — ground truth stain patterns.
            # Fake images: (fake, labels_dropped) — same conditioning the generator used.
            if self.proj_discriminator is not None and self.hparams.proj_disc_weight > 0:
                proj_real_out = self.proj_discriminator(ihc_for_disc, labels)
                proj_fake_out = self.proj_discriminator(fake_det_disc, labels_dropped)
                loss_proj_d = hinge_loss_d(proj_real_out, proj_fake_out)
                loss_d = loss_d + self.hparams.proj_disc_weight * loss_proj_d

            # R1 gradient penalty
            loss_r1 = torch.tensor(0.0, device=self.device)
            if self.global_step % self.hparams.r1_every == 0:
                with torch.amp.autocast('cuda', enabled=False):
                    if self.hparams.adversarial_weight > 0 and self.discriminator is not None:
                        real_input_r1 = ihc_for_disc.float().detach().requires_grad_(True)
                        for disc in [self.discriminator.disc_512]:
                            d_real = disc(real_input_r1)
                            grad_real = torch.autograd.grad(
                                outputs=d_real.sum(), inputs=real_input_r1,
                                create_graph=True,
                            )[0]
                            loss_r1 = loss_r1 + self.hparams.r1_weight * grad_real.pow(2).mean()
                    if self.uncond_discriminator is not None and (
                            self.hparams.uncond_disc_weight > 0 or self.hparams.feat_match_weight > 0):
                        ihc_r1 = ihc_for_disc.float().detach().requires_grad_(True)
                        d_real_uncond = self.uncond_discriminator(ihc_r1)
                        grad_uncond = torch.autograd.grad(
                            outputs=d_real_uncond.sum(), inputs=ihc_r1,
                            create_graph=True,
                        )[0]
                        loss_r1 = loss_r1 + self.hparams.r1_weight * grad_uncond.pow(2).mean()
                    if self.proj_discriminator is not None and self.hparams.proj_disc_weight > 0:
                        ihc_r1_proj = ihc_for_disc.float().detach().requires_grad_(True)
                        d_real_proj = self.proj_discriminator(ihc_r1_proj, labels)
                        grad_proj = torch.autograd.grad(
                            outputs=d_real_proj.sum(), inputs=ihc_r1_proj,
                            create_graph=True,
                        )[0]
                        loss_r1 = loss_r1 + self.hparams.r1_weight * grad_proj.pow(2).mean()
                loss_d = loss_d + loss_r1
                self.log('train/r1_penalty', loss_r1, prog_bar=False)

            opt_d.zero_grad()
            self.manual_backward(loss_d)
            if self.hparams.adversarial_weight > 0:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            if self.crop_discriminator is not None:
                torch.nn.utils.clip_grad_norm_(self.crop_discriminator.parameters(), 1.0)
            if self.uncond_discriminator is not None:
                torch.nn.utils.clip_grad_norm_(self.uncond_discriminator.parameters(), 1.0)
            if self.proj_discriminator is not None:
                torch.nn.utils.clip_grad_norm_(self.proj_discriminator.parameters(), 1.0)
            opt_d.step()

        # Logging
        self.log('train/loss_g', loss_g, prog_bar=True)
        self.log('train/loss_d', loss_d, prog_bar=True)
        self.log('train/lpips', loss_lpips, prog_bar=True)
        self.log('train/adversarial', loss_adv, prog_bar=False)
        self.log('train/lr_scale', lr_scale, prog_bar=False)
        if self.crop_discriminator is not None:
            self.log('train/crop_adv_g', loss_crop_adv, prog_bar=False)
            self.log('train/crop_adv_d', loss_crop_d, prog_bar=False)
        if self.uncond_discriminator is not None:
            self.log('train/uncond_adv_g', loss_uncond_adv, prog_bar=False)
            self.log('train/uncond_adv_d', loss_uncond_d, prog_bar=False)
        if self.proj_discriminator is not None:
            self.log('train/proj_adv_g', loss_proj_adv, prog_bar=False)
            self.log('train/proj_adv_d', loss_proj_d, prog_bar=False)
        if self.hparams.feat_match_weight > 0:
            self.log('train/feat_match', loss_feat_match, prog_bar=False)

    def on_validation_epoch_start(self):
        # Pick a random batch index for the second sample grid
        n_val_batches = max(1, len(self.trainer.val_dataloaders))
        self._random_val_batch_idx = torch.randint(1, max(2, n_val_batches), (1,)).item()
        # Per-label sample collectors (for multi-stain visual grids)
        self._val_per_label_samples = {}

    def _log_sample_grid(self, he, her2_01, gen_01, key):
        """Log H&E | Real | Gen grid to wandb."""
        n = min(4, len(he))
        he_01 = ((he[:n].cpu() + 1) / 2).clamp(0, 1)
        grid_images = []
        for i in range(n):
            grid_images.extend([
                he_01[i],
                her2_01[i].cpu(),
                gen_01[i].cpu(),
            ])
        grid = torchvision.utils.make_grid(grid_images, nrow=3, padding=2)
        if self.logger:
            self.logger.experiment.log({
                key: [wandb.Image(grid, caption='H&E | Real | Gen')],
                'global_step': self.global_step,
            })

    def validation_step(self, batch, batch_idx):
        he_rgb, ihc_rgb, he_h_map, ihc_h_map, he_e_map, labels, fnames = batch

        # On-the-fly UNI extraction
        if self._uni_extract_on_the_fly:
            uni_sub_crops = self._prepare_uni_sub_crops_from_tensor(he_rgb)
            uni = self._extract_uni_from_sub_crops(uni_sub_crops)
        else:
            raise ValueError("UNI features must be provided or extracted on-the-fly.")

        if self.hparams.disable_uni:
            uni = torch.zeros_like(uni)

        if self.hparams.disable_class:
            labels = torch.full_like(labels, self.hparams.null_class)

        # Use EMA generator (always Case B: H&E H-map input at inference)
        with torch.no_grad():
            generated = self.generator_ema(he_h_map, uni, labels, e_maps=he_e_map)

        # LPIPS (4x downsample: 128 for 512, 256 for 1024)
        lpips_size = self.hparams.image_size // 4
        gen_lpips = F.interpolate(generated, size=lpips_size, mode='bilinear', align_corners=False)
        ihc_lpips = F.interpolate(ihc_rgb, size=lpips_size, mode='bilinear', align_corners=False)
        lpips_val = self.lpips_fn(gen_lpips, ihc_lpips).mean()

        # SSIM
        gen_01 = ((generated + 1) / 2).clamp(0, 1)
        her2_01 = ((ihc_rgb + 1) / 2).clamp(0, 1)
        from torchmetrics.functional.image import structural_similarity_index_measure
        ssim_val = structural_similarity_index_measure(gen_01, her2_01, data_range=1.0)

        # DAB MAE (canonical: mean of top-10%)
        dab_gen = self.dab_extractor.extract_dab_intensity(generated.float().cpu(), normalize="none")
        dab_real = self.dab_extractor.extract_dab_intensity(ihc_rgb.float().cpu(), normalize="none")

        def p90_score(dab):
            flat = dab.flatten()
            p90 = torch.quantile(flat, 0.9)
            mask = flat >= p90
            return flat[mask].mean().item() if mask.sum() > 0 else flat.mean().item()

        dab_mae = sum(
            abs(p90_score(dab_gen[i]) - p90_score(dab_real[i]))
            for i in range(len(dab_gen))
        ) / len(dab_gen)

        self.log('val/lpips', lpips_val, prog_bar=True, sync_dist=True)
        self.log('val/ssim', ssim_val, prog_bar=True, sync_dist=True)
        self.log('val/dab_mae', dab_mae, prog_bar=True, sync_dist=True)

        # Collect per-label samples for visual grids (multi-stain only)
        if hasattr(self, '_val_per_label_samples'):
            for i in range(len(labels)):
                lbl = labels[i].item()
                if lbl == self.hparams.null_class:
                    continue
                if lbl not in self._val_per_label_samples:
                    self._val_per_label_samples[lbl] = {'he': [], 'real': [], 'gen': []}
                bucket = self._val_per_label_samples[lbl]
                if len(bucket['he']) < 4:
                    bucket['he'].append(he_rgb[i].cpu())
                    bucket['real'].append(her2_01[i].cpu())
                    bucket['gen'].append(gen_01[i].cpu())

        # Log sample grids: first batch (fixed) + one random batch
        if batch_idx == 0:
            self._log_sample_grid(he_rgb, her2_01, gen_01, 'val/samples_fixed')
        elif batch_idx == self._random_val_batch_idx:
            self._log_sample_grid(he_rgb, her2_01, gen_01, 'val/samples_random')

    def on_validation_epoch_end(self):
        """Log per-label sample grids if multiple labels are present."""
        if not hasattr(self, '_val_per_label_samples') or len(self._val_per_label_samples) <= 1:
            return

        label_names = getattr(self.hparams, 'label_names', None)

        for lbl, bucket in sorted(self._val_per_label_samples.items()):
            if not bucket['he'] or not self.logger:
                continue
            name = label_names[lbl] if label_names and lbl < len(label_names) else str(lbl)
            self._log_sample_grid(
                torch.stack(bucket['he']),
                torch.stack(bucket['real']),
                torch.stack(bucket['gen']),
                f'val/samples_{name}',
            )

        self._val_per_label_samples = {}

    @torch.no_grad()
    def generate(self, he_h_maps, uni_features, labels, e_maps=None,
                 num_inference_steps=None, guidance_scale=1.0, seed=None):
        """Generate IHC images from H-map (Hematoxylin channel) input.

        Args:
            he_h_maps: [B, 1, H, H] Hematoxylin channel extracted from H&E.
                       NOT raw H&E RGB — the generator encoder expects 1-channel input.
            uni_features: [B, N, 1024] where N=16 (4x4 CLS) or N=1024 (32x32 patch)
            labels: [B] class/stain labels
            num_inference_steps: ignored (single forward pass)
            guidance_scale: CFG scale (1.0 = no guidance)
            seed: random seed (for reproducibility, though model is deterministic)
        """
        if seed is not None:
            torch.manual_seed(seed)

        gen = self.generator_ema if hasattr(self, 'generator_ema') else self.generator

        if guidance_scale <= 1.0:
            return gen(he_h_maps, uni_features, labels, e_maps=e_maps)

        # Classifier-free guidance
        null_labels = torch.full_like(labels, self.null_class)

        output_cond = gen(he_h_maps, uni_features, labels, e_maps=e_maps)
        output_uncond = gen(he_h_maps, uni_features, null_labels, e_maps=e_maps)

        output = output_uncond + guidance_scale * (output_cond - output_uncond)
        return output.clamp(-1, 1)
