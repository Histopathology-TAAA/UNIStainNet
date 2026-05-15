# UNIStainNet — Architecture Deep Dive and Loss/Metric Report

This document summarizes the implementation details of UNIStainNet in this repository, with a particular focus on the Cross-Attention implementation (where it sits in the generator, exact tensor flows, and how UNI tokens are produced and consumed). It also enumerates all losses and evaluation metrics used in training and validation, listing the code locations, the purpose of each loss/metric, which training case it applies to (A = aligned IHC supervision, B = misaligned H&E supervision, Both), the hyperparameter name used for its weight and its default, and the W&B/trainer logging key used in code.

Contents
- **Architecture Overview**
- **Cross-Attention: Implementation & Tensor Shapes**
- **UNI Feature Extraction & Processors**
- **Where Cross-Attention Is Applied in the Decoder**
- **Eosin & Edge Encoders (where they inject signals)**
- **Discriminators (multi-scale, projection)**
- **Losses — detailed list**
- **Validation metrics — detailed list**
- **References to source files**

---

**Architecture Overview**
- Generator: `SPADEUNetGenerator` — see [src/models/generator.py](src/models/generator.py#L1-L40) and full implementation at [src/models/generator.py](src/models/generator.py#L1-L500).
  - Encoder processes 1-channel H-map input through progressively downsampling convs to a 16×16 bottleneck.
  - Bottleneck includes `ResBlock` and `SelfAttention` (global self-attention at bottleneck) — see [src/models/blocks.py](src/models/blocks.py#L1-L120).
  - Decoder upsamples in stages (16→32→64→128→256→512) and applies Cross-Attention conditioning at multiple decoder levels.
- UNI-conditioning: semantics come from a frozen UNI ViT-L/16 model (MahmoodLab/uni) and are provided to the generator as token arrays ([B, N, 1024]) produced either off-line or on-the-fly by `UNIStainNetTrainer._extract_uni_from_sub_crops()` — see [src/models/trainer.py](src/models/trainer.py#L340-L430).

**Cross-Attention: Implementation & Tensor Shapes**
- Cross-Attention module is implemented in `src/models/blocks.py` as `CrossAttention` (see [src/models/blocks.py](src/models/blocks.py#L60-L170)).
- Key characteristics (exact implementation):
  - Constructor: `CrossAttention(channels, uni_dim=1024, heads=4)` (default heads=4). The `channels` argument equals the decoder feature channels at that level (e.g., 512 for D5, 256 for D4, 128 for D3, 64 for D2, 64 for D1 when present).
  - Internals:
    - `norm = GroupNorm(32, channels)` applied to spatial features `x` before Q projection.
    - `token_norm = LayerNorm(uni_dim)` applied to `uni_tokens` (per-token normalization of UNI features).
    - `q_proj`: Conv2d( channels -> channels, kernel=1 ) producing Q from spatial features.
    - `k_proj`, `v_proj`: Linear( uni_dim -> channels ) applied to each UNI token to get K and V.
    - Heads split: `head_dim = channels // heads`; requires `channels % heads == 0`.
  - Forward: `forward(x, uni_tokens)` where
    - `x`: [B, C, H, W]
    - `uni_tokens`: either [B, N, D] (token list) or [B, D, H, W] (spatial map) — code accepts both.
    - If `uni_tokens` is 4D, it is permuted and reshaped to [B, N, D].
    - `uni_tokens` is normalized with LayerNorm.
    - Q flow: `q = q_proj(norm(x))` → reshape to [B, heads, head_dim, H*W] → permute to [B, heads, HW, head_dim].
    - K flow: `k = k_proj(uni)` → reshape to [B, N, heads, head_dim] → permute to [B, heads, head_dim, N].
    - V flow: `v = v_proj(uni)` → reshape to [B, N, heads, head_dim] → permute to [B, heads, N, head_dim].
    - Attention logits: `attn = (q @ k) * scale` where `scale = head_dim ** -0.5`, giving [B, heads, HW, N]. Softmax over N (`dim=-1`).
    - Attended values: `out = attn @ v` → [B, heads, HW, head_dim] → permute/recombine → [B, C, H, W].
    - Final projection: `out_proj` Conv2d( channels -> channels, 1 ), then residual add: `return x + out_proj(out)`.

- Important notes on semantics and invariances:
  - Q comes from local spatial generator features (querying what semantic tokens say about each spatial position).
  - K/V come from UNI patch tokens (semantic descriptors). This makes Cross-Attention a way to inject semantic guidance measured per-generator spatial position.
  - Because K/V are token-based (no spatial restriction), the attention allows each spatial position to attend to any semantic token.

**Where Cross-Attention Is Applied in the Decoder**
- Cross-attention modules are created and applied at these decoder levels inside `SPADEUNetGenerator` (see [src/models/generator.py](src/models/generator.py#L120-L260)):
  - `dec5_attn = CrossAttention(512, uni_dim=uni_dim, heads=4)` — used after D5 conv (32×32) (see generator forward at D5: around [src/models/generator.py](src/models/generator.py#L170-L200)).
  - `dec4_attn = CrossAttention(256, uni_dim=uni_dim, heads=4)` — used at 64×64.
  - `dec3_attn = CrossAttention(128, uni_dim=uni_dim, heads=4)` — used at 128×128.
  - `dec2_attn = CrossAttention(64, uni_dim=uni_dim, heads=4)` — used at 256×256.
  - `dec1_attn` (optional, image_size==1024) `CrossAttention(64, ...)` — used at 512×512 when present.

- Each decoder stage does: upsample -> concatenate skip(s) -> conv -> `decX_attn(x, uni_features)` -> activation. Thus attention is applied after the decoding conv at that scale and before upsampling to the next scale.

**Where UNI Tokens Originate and How They Are Prepared**
- Trainer on-the-fly extraction path (default in this repo): `UNIStainNetTrainer._prepare_uni_sub_crops_from_tensor()` slices the input H&E RGB into a 4×4 grid of sub-crops, resizes each to 224×224 and normalizes to ImageNet stats (see [src/models/trainer.py](src/models/trainer.py#L430-L520)). Then `_extract_uni_from_sub_crops()` passes these through the external UNI model (`timm` MahmoodLab/uni) to extract patch tokens, then interleaves patch tokens into a larger spatial grid and average-pools (adaptive) to the requested `uni_spatial_pool_size` (default 32) and finally returns `[B, S*S, 1024]` tokens where S is `uni_spatial_pool_size` — see [src/models/trainer.py](src/models/trainer.py#L340-L430).
- Alternatively the repository contains `UNIFeatureProcessor` and `UNIFeatureProcessorHighRes` (see [src/models/uni_processor.py](src/models/uni_processor.py#L1-L320)).
  - `UNIFeatureProcessor` expects CLS-level tokens (16 tokens, 4×4 grid) and expands them with learned transposed convs to multi-scale spatial maps (16→32→64→128→256), returning a dict of maps at those resolutions.
  - `UNIFeatureProcessorHighRes` expects patch tokens (1024 tokens, e.g. 32×32 grid), projects them and produces real 32×32 → 16/64/128/256 (and optionally 512) maps by convolutional processing.
- In this codebase the generator receives `uni_features` tokens and uses CrossAttention that accepts token arrays directly. The `uni_processor` modules are present for SPADE or alternate conditioning pathways (the SPADEBlock code accepts `uni_spatial` maps), but primary semantic injection used by this branch is the Cross-Attention token path.

**Eosin & Edge Encoders**
- `EosinEncoder` (see [src/models/edge_encoder.py]) is used to compress H&E Eosin channel into bottleneck membrane features (16×16) and optionally multi-scale 32×32 features injected into D5. Generator performs `self.eosin_proj` to combine Eosin features at bottleneck and optional `eosin_proj_32` at D5.
- Edge encoder (EdgeEncoder / MultiScaleEdgeEncoder) extracts structural features from H&E and supplies decoder skip-injections at multiple scales. The decoder concatenates `edge_maps` at relevant sizes to the skip tensors before convolution.

**Discriminators**
- `PatchDiscriminator`: patch-level PatchGAN with spectral norm; returns logits and (optionally) intermediate features for feature matching — see [src/models/discriminator.py](src/models/discriminator.py#L1-L120).
- `ProjectionDiscriminator`: stain-conditioned discriminator that projects a stain embedding into the penultimate feature channels and adds it to logits — used when `proj_disc_weight > 0` (see [src/models/discriminator.py](src/models/discriminator.py#L120-L320)).
- `MultiScaleDiscriminator` wraps two PatchDiscriminators (512 and 256 scales) — used when adversarial training is enabled.

---

**Losses — enumerated with code references, default weight hyperparameter, case applicability, and W&B / trainer log name**
(Where available, default values come from `UNIStainNetTrainer.__init__` argument defaults.)

Notes on "cases": the training loop randomly picks `use_aligned = rand() > case_b_prob`. If `use_aligned` is True → Case A (aligned) — generator input is IHC H-map and paired IHC RGB target is used for full-res L1/LPIPS and other aligned losses. If False → Case B (misaligned) — generator input is H&E H-map and misalignment-aware losses (spectral/LPIPS at downsample) are used. `case_b_prob` default is 0.5 (50/50) unless set otherwise.

- train/l1_fullres
  - Purpose: pixel-level L1 loss at full resolution for aligned samples (Case A). Encourages color-correct pixels when exact alignment is available.
  - Hyperparam: `l1_fullres_weight` (default 1.0)
  - Case: A only (applied when `use_aligned` True)
  - Logged under: `train/l1_fullres` (see [src/models/trainer.py](src/models/trainer.py#L820-L832))

- train/lpips_fullres
  - Purpose: perceptual similarity (LPIPS) at full resolution for aligned samples.
  - Hyperparam: `lpips_fullres_weight` (default 1.0)
  - Case: A only
  - Logged under: `train/lpips_fullres`

- train/spectral_misalign
  - Purpose: FFT-based high-frequency spectral loss used as a translation-invariant alternative in misaligned (Case B) mode. It compares log power spectra (focus on high-frequency band).
  - Hyperparam: `l1_lowres_weight` (the code uses this slot for spectral loss when `l1_lowres_weight>0`)
  - Default: `l1_lowres_weight=0.0` (off by default); when enabled configured by user.
  - Case: B only
  - Logged under: `train/spectral_misalign` (see [src/models/trainer.py](src/models/trainer.py#L855-L868))

- train/lpips (and train/lpips_fine)
  - Purpose: LPIPS perceptual loss at reduced resolution for Case B (misaligned) training; `lpips_weight` default=1.0 applied to main LPIPS at size image_size//4; `lpips_256_weight` default=0.5 is used optionally for half-resolution LPIPS (image_size//2).
  - Hyperparams: `lpips_weight` (default 1.0), `lpips_256_weight` (default 0.5), also `lpips_512_weight` present but default 0.0.
  - Case: B (misaligned) primarily — though `lpips_fullres` is Case A.
  - Logged under: `train/lpips` (fast-moving main scalar) and `train/lpips_fine` for the 256 LPIPS if used.

- train/dab_intensity
  - Purpose: Top-10% DAB intensity matching (vectorized batched top-10% mean). Matches the high-intensity DAB scoring between generated and ground truth.
  - Hyperparam: `dab_intensity_weight` (default 0.1)
  - Case: Both (applied regardless of `use_aligned`)
  - Logged under: `train/dab_intensity` (see [src/models/trainer.py](src/models/trainer.py#L904-L912))

- train/dab_histo
  - Purpose: Wasserstein-1 (sorted-L1) between DAB OD histograms (distribution-shape matching). Alignment-free.
  - Hyperparam: `dab_histo_weight` (default 0.0)
  - Case: Both
  - Logged under: `train/dab_histo` (see [src/models/trainer.py](src/models/trainer.py#L918-L924))

- train/dab_block
  - Purpose: per-block (dab_block_size) DAB mean matching — enforces spatial placement of DAB to match ground truth blocks (spatial supervisor).
  - Hyperparam: `dab_block_weight` (default 0.0), `dab_block_size` default 32
  - Case: A only (requires pixel alignment)
  - Logged under: `train/dab_block` (see [src/models/trainer.py](src/models/trainer.py#L926-L934))

- train/dab_sparsity
  - Purpose: hinge penalty when generated DAB mean > real DAB mean + margin (prevents over-staining). Alignment-free.
  - Hyperparam: `dab_sparsity_weight` (default 0.0), margin: `dab_sparsity_margin` default 0.05
  - Case: Both
  - Logged under: `train/dab_sparsity` (see [src/models/trainer.py](src/models/trainer.py#L940-L948))

- train/dab_contrast
  - Purpose: class-ordering hinge loss to encourage class-wise ordering (e.g., 3+ > 2+ > 1+ > 0). Uses label groups and pairwise margins.
  - Hyperparam: `dab_contrast_weight` (default 0.05)
  - Case: Both (uses `labels_dropped` to exclude CFG-dropped nulls)
  - Logged under: `train/dab_contrast` (see [src/models/trainer.py](src/models/trainer.py#L952-L959))

- train/edge_loss
  - Purpose: Fourier/FFT high-frequency spectral L1 to encourage boundary sharpness (applied as a separate `edge_weight`).
  - Hyperparam: `edge_weight` (default 0.0)
  - Case: (Used when configured) — the implementation is alignment-aware but is used where appropriate.
  - Logged under: `train/edge_loss` (see [src/models/trainer.py](src/models/trainer.py#L964-L973))

- train/dab_sharpness
  - Purpose: two-component DAB sharpness (1) mean Sobel gradient magnitude match and (2) sorted patch-variance L1 (local variance distribution). Encourages membrane-localized DAB rather than diffuse brown wash.
  - Hyperparam: `dab_sharpness_weight` (default 0.0)
  - Case: Both (alignment-free components included)
  - Logged under: `train/dab_sharpness` (see [src/models/trainer.py](src/models/trainer.py#L977-L986))

- train/gram_style
  - Purpose: Gram-matrix style loss using a VGG feature extractor to match texture statistics (alignment-invariant texture descriptor).
  - Hyperparam: `gram_style_weight` (default 0.0)
  - Case: Both (alignment-free)
  - Logged under: `train/gram_style` (see [src/models/trainer.py](src/models/trainer.py#L990-L998))

- train/ihc_edge
  - Purpose: IHC Sobel edge loss (Sobel L1) comparing generated edges to real IHC edges (Case A only). Strengthens boundary geometry where aligned ground truth is available.
  - Hyperparam: `ihc_edge_weight` (default 0.0)
  - Case: A only
  - Logged under: `train/ihc_edge` (see [src/models/trainer.py](src/models/trainer.py#L1004-L1010))

- train/he_edge
  - Purpose: H&E edge structure preservation — used in Case B to ensure generated images preserve structures seen in H&E input.
  - Hyperparam: `he_edge_weight` (default 0.0)
  - Case: B only (the code guards this with `not use_aligned`)
  - Logged under: `train/he_edge` (see [src/models/trainer.py](src/models/trainer.py#L1016-L1022))

- train/bg_white
  - Purpose: background white loss pushing background regions (as estimated from H&E brightness) toward white in generated IHC.
  - Hyperparam: `bg_white_weight` (default 0.0), `bg_threshold` default 0.85
  - Case: Both (applies per-image)
  - Logged under: `train/bg_white` (see [src/models/trainer.py](src/models/trainer.py#L1026-L1036))

- train/patchnce
  - Purpose: PatchNCE contrastive loss between H&E input and generated output encoder features. Alignment-aware at local patch level; used to preserve correspondences in features rather than exact pixels. Implemented in `src/models/losses.py::PatchNCELoss`.
  - Hyperparam: `patchnce_weight` (default 0.0), layer selection `patchnce_layers` default (2,3,4), `patchnce_n_patches` default 256, `patchnce_temperature` default 0.07
  - Case: Both (contrastive across encoder features); never uses GT IHC directly.
  - Logged under: `train/patchnce` (see [src/models/trainer.py](src/models/trainer.py#L1050-L1061))

- Adversarial losses / GAN losses (log keys)
  - train/adversarial (generator-side adversarial loss) — aggregated from `hinge_loss_g` across the outputs of `MultiScaleDiscriminator` if `adversarial_weight > 0`.
    - Hyperparam: `adversarial_weight` (default 1.0)
    - Case: Both (applies when `global_step >= adversarial_start_step`; default `adversarial_start_step=2000`)
    - Logged under: `train/adversarial` and `train/proj_adv_g` / `train/uncond_adv_g` / `train/crop_adv_g` depending on which discriminators are active.
  - Discriminator logs: `train/loss_d` (total discriminator loss), `train/crop_adv_d`, `train/uncond_adv_d`, `train/proj_adv_d`.
  - Projection discriminator (stain-conditioned) uses `proj_disc_weight` (default 0.0). When enabled it affects both generator and discriminator losses and is logged with `train/proj_adv_g` and `train/proj_adv_d`.
  - Unconditional discriminator weight: `uncond_disc_weight` (default 0.0) — logs `train/uncond_adv_g` and `train/uncond_adv_d`.
  - Crop discriminator (local crop adversarial): `crop_disc_weight` (default 0.0) logs `train/crop_adv_g` and `train/crop_adv_d`.

- train/feat_match
  - Purpose: feature matching loss (L1 between discriminator intermediate features) — used to stabilize GAN training and preserve texture statistics.
  - Hyperparam: `feat_match_weight` (default 0.0)
  - Case: Case A only (the trainer calls FM only when `use_aligned` is True)
  - Logged under: `train/feat_match` (see [src/models/trainer.py](src/models/trainer.py#L1049-L1058))

- train/r1_penalty
  - Purpose: R1 gradient penalty computed on real images for discriminator regularization.
  - Hyperparam: `r1_weight` (default 10.0) and frequency `r1_every` (default 16)
  - Case: Both (applies to whichever discriminators are used) — logged at `train/r1_penalty`.

- Aggregate logs and scalar names
  - `train/loss_g`: full generator objective after summation of applicable terms.
  - `train/loss_d`: full discriminator objective (including r1 penalty when computed).
  - `train/lpips`: primary LPIPS scalar tracked per-step.
  - `train/lr_scale`: the warmup LR scale (linear warmup) for debugging.

---

**Validation metrics and logging keys**
All validation logging is performed in `UNIStainNetTrainer.validation_step` and `on_validation_epoch_end` — see [src/models/trainer.py](src/models/trainer.py#L1360-L1700).

- `val/lpips` (LPIPS measured at image_size//4) — logged with `self.log('val/lpips', ...)`
  - Purpose: perceptual similarity for validation; used as main validation loss metric.
- `val/ssim` — structural similarity index (torchmetrics) — logged as `val/ssim`.
- `val/dab_mae` — canonical DAB p90 mean absolute error (p90 per-image canonical) — logged as `val/dab_mae`.
- Visual grids: `val/samples_fixed`, `val/samples_random`, and `val/samples_{label}` — logged to wandb as image grids via `self.logger.experiment.log` inside `_log_sample_grid`.
  - Implemented to collect per-label sample grids for multi-stain logging and remote inspection.

**Implementation references (key files)**
- Cross-Attention and other low-level blocks: [src/models/blocks.py](src/models/blocks.py#L1-L200)
- Generator: [src/models/generator.py](src/models/generator.py#L1-L500)
- UNI processors: [src/models/uni_processor.py](src/models/uni_processor.py#L1-L320)
- Trainer (losses, logging, GAN loop, UNI extraction, CFG dropout): [src/models/trainer.py](src/models/trainer.py#L1-L2000)
- Loss helpers: [src/models/losses.py](src/models/losses.py#L1-L260)
- Discriminators & GAN losses: [src/models/discriminator.py](src/models/discriminator.py#L1-L420)
- DAB extraction: [src/utils/dab.py](src/utils/dab.py#L1-L220)
- Metrics collection utilities: [src/utils/metrics.py](src/utils/metrics.py#L1-L420)

---

**Implementation notes / gotchas observed in the code**
- Cross-Attention expects decoder `channels` divisible by `heads` (default 4). If you change decoder channels or heads, ensure divisibility.
- `uni_tokens` may be either token arrays [B,N,D] or spatial maps [B,D,H,W]; `CrossAttention.forward` accepts both and reshapes spatial maps into token lists.
- UNI patch extraction (on-the-fly) builds a 56×56 interim grid of patch tokens when extracting from 4×4 sub-crops (4 sub-crops × 14 patches per side) then pools to `uni_spatial_pool_size` (default 32). See `_extract_uni_from_sub_crops` in [src/models/trainer.py](src/models/trainer.py#L360-L430).
- There are two semantic conditioning pathways present: Cross-Attention (token K/V) and SPADE-style spatial modulation (`SPADEBlock`) that can consume `uni_spatial` maps. This branch primarily uses cross-attention — `SPADEBlock` remains in the codebase for experiments or alternate modes.
- Eosin multi-scale injection (bottleneck 16×16 + optional 32×32 D5) is available behind `eosin_multi_scale` and `use_eosin_encoder`. Enabling it changes state_dict keys and is incompatible with older checkpoints.
- The trainer registers a dummy parameter when all discriminators are disabled so the Adam optimizer can be created with a non-empty param group. This is intentional for safety in ablation/sanity check runs.
- CFG dropout is implemented vectorized in `_apply_cfg_dropout` by zeroing UNI vectors or replacing labels with `null_class` for unlabeled guidance.

---

**Quick map: where to read the core implementations**
- CrossAttention implementation: [src/models/blocks.py](src/models/blocks.py#L60-L170)
- CrossAttention usage in Generator (which decoder stages): [src/models/generator.py](src/models/generator.py#L150-L235)
- UNI extraction (on-the-fly): [src/models/trainer.py](src/models/trainer.py#L340-L430)
- DAB extraction (color deconvolution): [src/utils/dab.py](src/utils/dab.py#L1-L220)
- All loss functions and PatchNCE: [src/models/losses.py](src/models/losses.py#L1-L260)
- Training loop and exact logging keys: [src/models/trainer.py](src/models/trainer.py#L780-L1200) and validation at [src/models/trainer.py](src/models/trainer.py#L1360-L1700)

---

If you'd like, I can now:
- Generate an annotated call graph showing exactly which functions call `CrossAttention` and which call `SPADEBlock` (auto-extracted). 
- Produce a compact PDF or notebook version of this report.
- Run a static analysis to produce the list of all hyperparameters (with defaults) and a CSV suitable for W&B config import.

Tell me which of the above you'd like next, or if you want deeper per-line annotations embedded into the source files (read-only comments saved separately).