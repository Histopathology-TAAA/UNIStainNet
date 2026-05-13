# Architecture & Training Changelog

Each version documents *why* the change was made, *what* changed, and *where* in the codebase.
Newest version last. Do not delete older entries.

---

## V0 — Initial Destaining Refactor (baseline)

**Branch:** `exp-destaining-v1-attention`
**Motivation:** Adapt the original BCI/MIST codebase to a structure-semantic decoupled
approach: strip stain color from H&E (Hematoxylin channel only), inject biological
meaning via frozen UNI ViT-L/16 tokens.

### Dataset
- Added paired loading of RGB images and 1-channel H-maps per stain with the new
  folder layout: `trainA`, `trainA-H`, `trainB`, `trainB-H`, `valA`, `valA-H`, `valB`, `valB-H`.
- Ensured paired crops and spatial augmentations are applied identically across H&E RGB,
  IHC RGB, H&E H-map, and IHC H-map.
- Normalized H-maps as 1-channel tensors using `mean=[0.5], std=[0.5]` → range `[-1, 1]`.

### Architecture
- Decoupled structure and semantics: the UNet encoder now consumes 1-channel structural
  inputs (H-maps) instead of RGB.
- Replaced SPADE conditioning blocks with 4-head Cross-Attention blocks that use GroupNorm;
  Q from spatial features, K/V from UNI tokens.
- Added residual connections around Cross-Attention to avoid bottlenecks.

### Training Logic
- Implemented 50/50 mixed-domain routing per batch:
  - Case A (aligned): generator input = IHC H-map; losses = full-res L1 + full-res LPIPS vs IHC RGB.
  - Case B (misaligned): generator input = H&E H-map; losses = L1 @ 64×64 + LPIPS @ 128 & 256.
- UNI semantic tokens always extracted from H&E RGB regardless of routing mode.
- Validation always uses H&E H-map → IHC RGB (inference-time domain).

### Discriminator
- Kept the discriminator unconditional: evaluates IHC image only (real vs generated),
  not concatenated with H&E inputs.
- Feature matching and R1 penalty computed from unconditional discriminator features.

### Sanity Check Script
- `scripts/debug/sanity_check_mist_dummy.py`: bypasses UNI model download, runs one
  train + val step with dummy tensors.

---

## V1 — Bug Fixes, Structural Corrections & Domain Split Refinement

**Motivation:** Code audit triggered by first 20 epochs of training revealed multiple
semantic and technical bugs. Key observation: model was "over-expressive" on non-tumor
patches (light brown wash) and producing a nuclear shortcut for HER2 (nuclei stained
but membranes missed). Several bugs were actively harming training.

### Bug Fixes

| Bug | Root Cause | Fix | File |
|-----|-----------|-----|------|
| `KeyError: PosixPath(...)` in dataset | `Counter(s[2] for s in samples)` — index 2 is a `PosixPath`, not the stain label | Changed to `s[4]` (stain label is at index 4) | `src/data/mist_dataset.py` |
| `generate()` crashed with 3-ch input | Docstring said H&E RGB; generator expected 1-ch H-map | Renamed param `he_images` → `he_h_maps`, fixed docstring | `src/models/trainer.py` |
| DAB contrast used wrong labels | `compute_dab_contrast_loss(generated, labels)` — but CFG-dropped samples have null_class label, causing wrong class-ordering gradients | Changed to `labels_dropped` | `src/models/trainer.py` |
| HE edge loss active in Case A | In Case A the H-map and H&E are from different tissue sections → conflicting gradient | Added `and not use_aligned` guard | `src/models/trainer.py` |
| MultiScaleDiscriminator wasted VRAM when `adversarial_weight=0` | Always instantiated unconditionally | Wrapped in `if adversarial_weight > 0` | `src/models/trainer.py` |
| FM loss included final logit layer | `d_feats_fake` included the last 1-ch output, which compares raw scores not texture | Added `[:-1]` slice in `feature_matching_loss` | `src/models/discriminator.py` |
| Edge encoder v2 expected 5-ch input | Older code did `h_maps.repeat(1,3,1,1)` → fake RGB + gradients (3+2=5 ch) | Refactored `MultiScaleEdgeEncoder` to native 1-ch input `[h_map, gx, gy]` (3ch); removed repeat hack | `src/models/edge_encoder.py`, `src/models/generator.py` |
| Empty discriminator optimizer | When `adversarial_weight=uncond_disc_weight=crop_disc_weight=0`, `disc_params=[]` → `Adam` crash | Added dummy parameter: `self._disc_dummy = nn.Parameter(torch.zeros(1))` | `src/models/trainer.py` |
| Lightning AI `$HOME` path doubled | Launch script used `$HOME/teamspace/studios/this_studio` but on Lightning AI `$HOME` IS that path | Fixed to `STUDIO_ROOT="$HOME"` | `scripts/train/launch_destaining.sh` |

### Structural Improvements

- **FFT spectral loss replaces L1 @ 64×64 in Case B.** L1 @ 64×64 in Case B was
  translation-sensitive — the ~30px slice misalignment caused the gradient to penalize
  the generator for something it cannot fix. Replaced with `compute_edge_loss()` which
  computes L1 on log-FFT magnitudes (high-frequency band). Frequency content is
  translation-invariant by construction: shifting an image does not change its power spectrum.
  (`src/models/trainer.py`)

- **Feature matching restricted to Case A.** In Case B, the discriminator's spatial feature
  maps encode where nuclei are relative to the 30px shifted real IHC. FM loss told the
  generator "your nuclei are in the wrong location" — a gradient it cannot satisfy. The
  model responded by diffusing DAB broadly (brown wash). Adding `and use_aligned` guard
  eliminates this. (`src/models/trainer.py`)

- **Domain split made configurable.** `case_b_prob` param (default changed from implicit
  0.5 to 0.25): 75% of steps now use aligned supervision (Case A), giving higher-quality
  gradients. CLI: `--case_b_prob`. Do not go below 0.20 (inference domain shock risk).
  (`src/models/trainer.py`, `scripts/train/train_mist.py`)

---

## V2 — Eosin Bottleneck Injection

**Motivation:** After 20 epochs, HER2 output showed a *nuclear shortcut*: the model
correctly identified nuclei from the H-map but could not localize DAB to cell membranes
(the HER2 scoring criterion). Root cause: the Hematoxylin H-map encodes only nuclei;
membrane topology is exclusively in the Eosin channel, which the generator never saw.

**Solution:** Inject H&E Eosin channel at the bottleneck (16×16). At this resolution
the ~30px physical misalignment between consecutive tissue sections is <1px — the
injection is therefore misalignment-safe without any special loss design.

### New Component: `EosinEncoder`
**File:** `src/models/edge_encoder.py`
- 5-stage strided-conv downsampler: 512→256→128→64→32→16.
- Outputs `[B, out_channels, 16, 16]` (default `out_channels=64`).
- Takes H&E Eosin map `[B, 1, 512, 512]` in `[-1, 1]`.

### Generator Changes
**File:** `src/models/generator.py`
- Import: added `EosinEncoder` from `edge_encoder`.
- New `__init__` params: `use_eosin_encoder=False`, `eosin_out_ch=64`.
- New modules: `self.eosin_encoder = EosinEncoder(...)`, `self.eosin_proj = nn.Conv2d(512+eosin_out_ch, 512, 1)`.
- `forward` signature: added `e_maps=None`.
- Injection point: after bottleneck ResBlock+SelfAttn:
  ```python
  x = self.bottleneck(e5)                            # [B, 512, 16, 16]
  if self.eosin_encoder is not None and e_maps is not None:
      e_feat = self.eosin_encoder(e_maps)            # [B, 64, 16, 16]
      x = self.eosin_proj(torch.cat([x, e_feat], 1)) # [B, 512, 16, 16]
  ```

### Dataset Changes
**File:** `src/data/mist_dataset.py`
- Loads `trainA-E` / `valA-E` Eosin map directories (optional: prints warning + returns
  zeros if directories are absent — training continues without Eosin data).
- Sample tuple expanded: 5-tuple → 6-tuple (added `he_e_path | None` at index 4; `stain_label` now at index 5).
- `__getitem__` returns 7-tuple: `(he_rgb, ihc_rgb, he_h_map, ihc_h_map, he_e_map, label, filename)`.
- `_random_crop_quad` and `_apply_paired_augmentations_quad` extended with optional 5th image arg.
- `Counter` index corrected from `s[4]` → `s[5]` to match new tuple layout.

### Trainer Changes
**File:** `src/models/trainer.py`
- New params: `use_eosin_encoder=False`, `eosin_out_ch=64`.
- Wired to `SPADEUNetGenerator(...)`.
- `training_step` batch unpack: 6-tuple → 7-tuple (adds `he_e_map`).
- All generator calls updated: `self.generator(..., e_maps=he_e_map)`.
- `validation_step` updated: same 7-tuple unpack + `e_maps=he_e_map` in EMA generator call.
- `generate()` API extended: `e_maps=None` parameter (CFG guidance path also passes it).

### Supporting Files
- `scripts/debug/sanity_check_mist_dummy.py`: added `he_e_map` tensor to dummy batch,
  updated collate function, added `use_eosin_encoder=True` to trainer (full round-trip test).
- `scripts/train/train_mist.py`: added `--use_eosin_encoder` / `--no_use_eosin_encoder` flag
  (default True); wired `eosin_out_ch=64`.

### Hyperparameter Control
| Param | Default | Effect of changing |
|-------|---------|-------------------|
| `use_eosin_encoder` | `True` | Set `False` to disable entirely (no VRAM, no compute) |
| `eosin_out_ch` | `64` | Increase for more capacity (128), decrease to save memory |

---

## V3 — MLPA-Style Losses + Stain-Conditioned Projection Discriminator

**Motivation:** After V2, two gaps remained:
1. **DAB supervision is too coarse.** The `dab_intensity` loss matches only the top-10%
   mean — one scalar per image. A generator can fool this by producing a uniform 1+ stain
   while the real image has a bimodal 0/3+ distribution, or by placing DAB in the wrong
   spatial regions.
2. **The unconditional discriminator cannot enforce stain-specific staining patterns.**
   HER2 scores membrane staining; Ki67/ER/PR score nuclear staining. An unconditional
   discriminator just checks "does this look like IHC?" regardless of where the DAB is.
   Without stain-awareness, the generator gets no adversarial gradient for the HER2
   nuclear shortcut.

### New Losses

#### IHC-to-IHC Sobel Edge Loss (Case A only)
**File:** `src/models/trainer.py`
**Param:** `ihc_edge_weight` (default `0.1` in train_mist.py, `0.0` in trainer to allow disabling)
**W&B key:** `train/ihc_edge`

Applies existing `compute_he_edge_loss(generated, ihc_rgb)` in Case A — when generator input
is the IHC H-map, a pixel-aligned Sobel edge comparison against real IHC is valid.
Penalizes: blurry/wrong membrane and cell boundary structure in generated IHC.
Why Case A only: in Case B the IHC H-map and H&E have 30px tissue drift — Sobel edges
from different tissue sections would conflict with full-res supervision.

#### DAB Histogram Matching (both cases)
**File:** `src/models/trainer.py`, new method `compute_dab_histogram_loss`
**Param:** `dab_histo_weight` (default `0.3`)
**W&B key:** `train/dab_histo`

Wasserstein-1 distance between DAB optical density distributions:
1. Extract DAB OD via color deconvolution (Ruifrok & Johnston).
2. Clamp at 99th percentile of target (suppress non-specific background noise).
3. Sort both distributions independently and compute L1 (sorted-L1 = W1 distance).

Alignment-free: compares distribution shape, not spatial layout. Valid in both Case A and B.
Penalizes: uniform staining masquerading as the correct mean — catches bimodal distribution
errors (e.g., generating 1+ everywhere when real image has strong 0+/3+ bimodality).

#### Block-Level DAB Spatial Matching (Case A only)
**File:** `src/models/trainer.py`, new method `compute_dab_block_loss`
**Params:** `dab_block_weight` (default `0.2`), `dab_block_size` (default `32`)
**W&B key:** `train/dab_block`

`avg_pool2d(dab_map, kernel_size=block_size, stride=block_size)` → L1 between block DAB means.
For 512×512 images with block_size=32: 16×16 = 256 blocks per image.
Penalizes: wrong spatial distribution of DAB — forces the generator to put staining
*where* the ground truth has it, not just match global intensity/distribution.
Case A only (same misalignment argument as feature matching).

### New Discriminator: `ProjectionDiscriminator`
**File:** `src/models/discriminator.py` (new class)
**Trainer param:** `proj_disc_weight` (default `1.0` in train_mist.py, `0.0` in trainer)
**W&B keys:** `train/proj_adv_g`, `train/proj_adv_d`

Based on Miyato & Koyama (2018) "cGANs with Projection Discriminator."

Architecture: standard PatchGAN backbone (same depth as other discriminators) but the final
step replaces the 4×4 output conv with:
```
logits = Conv1x1(features)  +  (features · Linear(stain_embed(label))).sum(dim=1)
```
The stain embedding is projected to the feature channel dimension and inner-producted with
each spatial location's feature vector. This biases the discriminator's real/fake decision
based on the stain label:
- **HER2**: discriminator learns membrane-localized DAB is real; nuclear-only DAB is penalized.
- **Ki67 / ER / PR**: discriminator learns nuclear DAB is real.

Why not condition on H&E (like the existing conditional discriminator design)?
- The spatial misalignment between H&E and IHC would make the discriminator penalize
  correct outputs for the wrong reason (learned to flag the 30px drift, not realism).
- Conditioning on the stain label alone is misalignment-safe: the stain embedding carries
  stain-type information without any spatial correspondence requirement.

**Label protocol:**
- Generator adversarial step: `proj_disc(generated, labels_dropped)` — same labels the generator was conditioned on.
- Discriminator real step: `proj_disc(real_ihc, labels)` — true stain labels (never null_class on real data).
- Discriminator fake step: `proj_disc(fake, labels_dropped)` — consistent with how fake was generated.
- R1 gradient penalty: applied on real images with `labels` (true labels).

**To disable completely:** set `proj_disc_weight=0.0` — the discriminator is never instantiated
(zero VRAM, zero compute, zero change to any other loss).

### Trainer Wiring Summary
**File:** `src/models/trainer.py`
- 5 new `__init__` params: `ihc_edge_weight`, `dab_histo_weight`, `dab_block_weight`, `dab_block_size`, `proj_disc_weight`.
- `any_adv` flag updated to include `proj_disc_weight > 0`.
- `configure_optimizers`: proj_disc params added to disc optimizer.
- Grad clipping and R1 penalty extended to proj_disc.
- All new losses logged to W&B.

**File:** `scripts/train/train_mist.py` — active defaults:
```python
ihc_edge_weight=0.1
dab_histo_weight=0.3
dab_block_weight=0.2
dab_block_size=32
proj_disc_weight=1.0
```

### Hyperparameter Control
| Param | Default (train_mist.py) | Safe to disable | How |
|-------|------------------------|----------------|-----|
| `ihc_edge_weight` | `0.1` | Yes | Set `0.0` |
| `dab_histo_weight` | `0.3` | Yes | Set `0.0` |
| `dab_block_weight` | `0.2` | Yes | Set `0.0` |
| `dab_block_size` | `32` | — | Increase for coarser supervision |
| `proj_disc_weight` | `1.0` | Yes | Set `0.0` (discriminator not instantiated) |

---

## V4 — Weight Tightening (Option A) + Multi-Scale Eosin Injection (Option B)

**Motivation:** Epoch 24 analysis revealed three persisting problems:
1. **Ki67/ER/PR over-expressiveness**: Model paints too many nuclei DAB-positive. dab_block and dab_histo losses plateaued — additional direct mass constraint needed.
2. **HER2 nuclear shortcut unchanged**: proj_adv_g still oscillating -1.0 to +1.5 with no variance compression. Generator failing to satisfy stain-conditioned disc. Discriminator pressure needs to be doubled.
3. **Architectural limit**: Eosin bottleneck at 16×16 (32px/pixel) is too coarse to resolve individual cell membrane rings (~1–2px at 512px). A second injection at 32×32 (16px/pixel) would provide finer membrane topology to the first decoder level.

**Also observed:** val/ssim dropped sharply at step 52k and val/dab_mae regressed from 0.356 back to 0.362. Best checkpoint is approximately step 48k — resume from there.

### Option A — Weight Adjustments (Resume-compatible)

**Files:** `scripts/train/train_mist.py`, `src/models/trainer.py`

#### New: DAB Sparsity Hinge Loss
**File:** `src/models/trainer.py` — new method `compute_dab_sparsity_loss`
**Param:** `dab_sparsity_weight` (default `0.3`), `dab_sparsity_margin` (default `0.05`)
**W&B key:** `train/dab_sparsity`

Hinge penalty: `relu(gen_dab_mean - real_dab_mean - margin).mean()` per image.
Applied in both Case A and B (alignment-free: compares per-image means, not spatial positions).
Targets the Ki67/ER/PR mass problem directly: penalises only when the total DAB mass is
too high, not when it's correctly sparse or negative. Complements dab_histo (distribution
shape) and dab_block (spatial placement) with a direct mass ceiling.

#### Weight Changes
| Param | V3 default | V4 default | Reason |
|-------|-----------|-----------|--------|
| `dab_block_weight` | `0.2` | `0.5` | Plateau at V3 value; needs stronger spatial constraint |
| `proj_disc_weight` | `1.0` | `2.0` | HER2 proj_adv_g not converging; doubles discriminator pressure |
| `dab_sparsity_weight` | — | `0.3` | New; direct over-staining mass control |

**To resume from step-48k checkpoint with Option A weights:**
```bash
python scripts/train/train_mist.py \
    --data_dir /teamspace/studios/this_studio/data/MIST \
    --resume_from checkpoints/destaining_v1/mist_epoch022_step048000.ckpt \
    --wandb_name destaining_v1_optionA
```

### Option B — Multi-Scale Eosin Injection (New runs only)

**Files:** `src/models/edge_encoder.py`, `src/models/generator.py`, `src/models/trainer.py`, `scripts/train/train_mist.py`

**Architectural change:** A second Eosin injection is added at decoder level D5 (32×32)
in addition to the existing bottleneck injection (16×16). At 32×32 the ~30px tissue
misalignment is ~1–2px — still negligible. This gives the decoder a membrane signal at
finer resolution one level before the bottleneck forces coarsening.

#### EosinEncoder changes (`src/models/edge_encoder.py`)
- New param: `multi_scale=False` (backward compatible default).
- `multi_scale=False`: keeps `self.encoder = nn.Sequential(...)` — state_dict keys
  unchanged, all existing checkpoints continue to load.
- `multi_scale=True`: uses staged `self.stage1..stage5`. `forward` returns
  `(feat_16, feat_32)` — [B, out_ch, 16, 16], [B, 64, 32, 32].

#### Generator changes (`src/models/generator.py`)
- New param: `eosin_multi_scale=False`.
- When True: `EosinEncoder(multi_scale=True)`, new `self.eosin_proj_32 = nn.Conv2d(512+64, 512, 1)`.
- Injection point: after `dec5_act` (first decoder upsampling, before D4).

#### Trainer changes (`src/models/trainer.py`)
- New param: `eosin_multi_scale=False`, passed to generator.

#### CLI (`scripts/train/train_mist.py`)
```bash
# Enable Option B for a NEW run (do NOT combine with --resume_from):
python scripts/train/train_mist.py \
    --data_dir /teamspace/studios/this_studio/data/MIST \
    --eosin_multi_scale \
    --wandb_name destaining_v1_optionB
```

**WARNING**: `--eosin_multi_scale` changes EosinEncoder state_dict keys
(`stage1.0.weight` vs `encoder.0.weight`). Combining with `--resume_from` from
a V3 checkpoint will raise a key mismatch error. Only use for fresh runs.

### Hyperparameter Control (V4)
| Param | Default (train_mist.py) | Safe to disable | How |
|-------|------------------------|----------------|-----|
| `dab_block_weight` | `0.5` | Yes | Set `0.0` |
| `proj_disc_weight` | `2.0` | Yes | Set `0.0` (discriminator not instantiated) |
| `dab_sparsity_weight` | `0.3` | Yes | Set `0.0` |
| `dab_sparsity_margin` | `0.05` | — | Increase to allow more tolerance before penalty |
| `eosin_multi_scale` | `False` | — | `--eosin_multi_scale` to enable (new runs only) |
