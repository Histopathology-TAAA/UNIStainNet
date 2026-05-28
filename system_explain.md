# UNIStainNet: Complete System Explanation

**Last Updated:** May 26, 2026 | **Active Branch:** `exp-destaining-v1-attention` | **Current Version:** V4

This document provides a comprehensive explanation of the UNIStainNet project for any new LLM or developer joining the work. Read this to understand the complete system architecture, current state, training approach, and how to extend or debug the codebase.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [The Destaining Approach (Core Innovation)](#the-destaining-approach-core-innovation)
3. [Current Status: Version 4 (V4)](#current-status-version-4-v4)
4. [Architecture Deep Dive](#architecture-deep-dive)
5. [Training Loop & Loss Configuration](#training-loop--loss-configuration)
6. [Dataset & Data Flow](#dataset--data-flow)
7. [Key Components & File Guide](#key-components--file-guide)
8. [Practical Workflows](#practical-workflows)
9. [Common Pitfalls & Solutions](#common-pitfalls--solutions)
10. [Evaluation & Metrics](#evaluation--metrics)
11. [Development Guidelines](#development-guidelines)

---

## Project Overview

### What is UNIStainNet?

UNIStainNet is a **foundation-model-guided virtual staining network** that translates H&E (Hematoxylin & Eosin) histopathology images into IHC (Immunohistochemistry) stained images. A single 42M-parameter model generates multiple IHC stains (HER2, Ki67, ER, PR) from H&E input, conditioned on learned stain embeddings.

### Key Results (Paper Metrics)

| Dataset | FID ↓ | KID×1k ↓ | SSIM ↑ | Pearson-r ↑ | DAB KL ↓ |
|---------|-------|----------|--------|------------|----------|
| **MIST (unified 4-stain)** | 29-35 | 1.1-2.2 | 0.23-0.28 | 0.93-0.95 | 0.12-0.18 |
| **BCI (HER2 only)** | 34.6 | 6.5 | 0.541 | 0.867 | 0.482 |

### Main Contributions

1. **Dense UNI Spatial Conditioning**: 32×32 = 1,024 spatial tokens from frozen UNI ViT-L/16 (64× more than CLS-token approaches)
2. **Misalignment-Aware Training**: Consecutive tissue sections have ~30px drift; training handles this via mixed-domain routing
3. **Unified Multi-Stain Model**: Single model replaces 4 specialized models with 4× fewer parameters
4. **Structure-Semantic Decoupling**: H-map encoder for structure, Cross-Attention for semantics (novel destaining approach)

---

## The Destaining Approach (Core Innovation)

### The Problem This Solves

**Standard approach (original BCI/MIST codebase):**
- Input: H&E RGB → contains stain color (blue Hematoxylin, pink Eosin)
- Generator sees both stain and structure mixed together
- Hard to isolate what the model learns about tissue structure vs stain appearance

**Destaining approach (current branch):**
- Decouples structure from stain color
- Input: H-map (Hematoxylin channel only) = pure tissue structure, no stain color
- Semantic context: frozen UNI ViT-L/16 tokens (learned on pathology images)
- This forces the generator to reason explicitly: "Here is tissue structure. Here are biological semantics. Generate the target stain."

### How It Works (4-Step Pipeline)

```
1. DESTAIN H&E
   H&E RGB (512×512) → Color deconvolution (Ruifrok & Johnston)
   → H-map (1ch) + E-map (1ch) [pure tissue structure, no stain]

2. ENCODE STRUCTURE
   H-map (1ch) → UNet Encoder (progressively downsample to 16×16)
   → Bottleneck features [B, 512, 16, 16]

3. INJECT SEMANTICS
   H&E RGB (512×512) → UNI ViT-L/16 (frozen)
   → 1,024 patch tokens [B, 1024, 1024-dim]
   → Cross-Attention at every decoder level
   (Q = spatial features from decoder, K/V = UNI tokens)

4. RESTAIN
   Bottleneck features + Eosin injection (16×16)
   → Decoder (16→32→64→128→256→512)
   → At each level: conv → Cross-Attention → SPADE+FiLM → ReLU
   → Output: IHC RGB [B, 3, 512, 512]
```

### Why This Matters

1. **Structure-only supervision**: When training on aligned IHC H-maps (Case A), losses directly supervise tissue structure
2. **Translation-invariant training**: Consecutive tissue sections are misaligned by ~30px; destaining lets us use alignment-free losses
3. **Cleaner semantic injection**: UNI tokens carry biological meaning; separated from H-map structure
4. **Better membrane localization**: Eosin channel provides cell membrane context that pure H-map lacks

---

## Current Status: Version 4 (V4)

### What Changed Across Versions

| Version | Key Changes | Focus |
|---------|-------------|-------|
| **V0** | Initial destaining refactor + mixed-domain routing | Structure-semantic decoupling |
| **V1** | Bug fixes, FFT spectral loss for Case B, configurable domain split | Fixing oversaturation + structural issues |
| **V2** | Eosin bottleneck injection at 16×16 | Fixing HER2 nuclear shortcut |
| **V3** | DAB histogram + block losses, stain-conditioned discriminator | Adding stain-specific supervision |
| **V4** | Weight tightening (dab_block 0.2→0.5, proj_disc 1.0→2.0) + sparsity loss | Addressing Ki67/ER/PR over-staining |

### V4 Status (As of May 2026)

**Problem Observed at Epoch 24:**
- Ki67/ER/PR stains too "expressive" (over-painting nuclei)
- HER2 still shows nuclear shortcut (proj_adv_g oscillating wildly -1.0 to +1.5)
- Metric dips at step 52k: val/ssim dropped sharply, val/dab_mae regressed

**Solution in V4:**
1. Increased `dab_block_weight` from 0.2 → 0.5 (spatial DAB placement stronger)
2. Doubled `proj_disc_weight` from 1.0 → 2.0 (stain-conditioned realism pressure)
3. Added `dab_sparsity_loss` (direct hinge penalty on over-staining)
4. **Best checkpoint found at step ~48k** — resume from there for Option A

**Two Parallel Options Going Forward:**

| Option | Approach | Status | How to Use |
|--------|----------|--------|-----------|
| **A (Resume)** | Resume step-48k with tightened weights | Ready now | `--resume_from checkpoints/.../step048000.ckpt` |
| **B (Fresh)** | New run with multi-scale Eosin injection at D5 (32×32) | Experimental | `--eosin_multi_scale` (new runs only) |

**Current Best Checkpoint Metrics (Step 48k):**
- val/ssim: high
- val/dab_mae: 0.356
- val/lpips: stable
- Status: Before the regression at step 52k

---

## Architecture Deep Dive

### Component Hierarchy

```
SPADEUNetGenerator
├── H-Map Encoder (1ch input)
│   ├── enc0 (for 1024 only): 1024→512
│   ├── enc1-5: Progressive downsampling to 16×16 bottleneck
│   └── Bottleneck: ResBlock + SelfAttention + ResBlock
├── Eosin Encoder (optional)
│   └── Injects membrane topology at bottleneck (16×16)
│   └── Optional: second injection at D5 (32×32) with --eosin_multi_scale
├── Decoder (16→512 upsampling)
│   ├── D5 (16×32): conv → CrossAttention → SPADE+FiLM
│   ├── D4 (32→64): conv → CrossAttention → SPADE+FiLM
│   ├── D3 (64→128): conv → CrossAttention → SPADE+FiLM
│   ├── D2 (128→256): conv → CrossAttention → SPADE+FiLM
│   ├── D1 (256→512, for 1024 only): conv → CrossAttention → SPADE+FiLM
│   └── Final conv: 64ch → 3ch RGB output
├── Skip Connections (from encoder to decoder)
│   └── Concatenated before each decoder conv
├── UNI Processor
│   └── Converts patch tokens [B, 1024, 1024] → multi-scale spatial maps
└── Class Embedding
    └── Stain label → 64-dim vector (used by SPADE FiLM)
```

### Cross-Attention: The Semantic Injection Point

**Location:** At each decoder stage (D5, D4, D3, D2, D1 for 1024)

**Input & Output:**
- Input: spatial features `x` [B, C, H, W] + UNI tokens [B, N, 1024]
- Output: attended features [B, C, H, W] (same shape, added as residual to x)

**Computation (4-head attention, per stage):**

```
1. Normalize spatial features with GroupNorm(32, C)
2. Project spatial features to Q: [B, C, H, W] → [B, C, H, W] via Conv1x1
3. Normalize UNI tokens with LayerNorm(1024): [B, 1024, 1024]
4. Project tokens to K/V: Linear(1024 → C) separately
5. Reshape into multi-head form: Q=[B, heads, HW, head_dim], K=[B, heads, head_dim, N], V=[B, heads, N, head_dim]
6. Attention: A = softmax(QK^T / sqrt(d)) → [B, heads, HW, N]
7. Attended: out = A @ V → [B, heads, HW, head_dim] → reshape to [B, C, H, W]
8. Project and add residual: x = x + Conv1x1(out)
```

**Why This Matters:**
- Every spatial position can attend to all 1,024 UNI tokens
- UNI tokens capture biological semantics (trained on pathology images)
- Allows local spatial features to be modulated by global semantic context
- Residual path: attention can contribute a correction without dominating the skip-connection signal

### SPADE & FiLM Conditioning

**SPADE (Spatially-Adaptive Normalization):**
- Traditional SPADE: uses spatial maps to modulate normalization γ/β
- This implementation: uses UNI spatial maps [B, C, H, W] at each resolution
- Located after conv, before activation

**FiLM (Feature-wise Linear Modulation):**
- Stain embedding [B, 64] → γ, β scalars
- Applied to normalize features: y = γ * x + β
- Allows stain label to globally adjust feature statistics

**Interplay:**
- SPADE provides **spatial, semantic conditioning** (UNI tokens)
- FiLM provides **global stain conditioning** (stain embedding)
- Together they modulate the decoder to generate stain-specific outputs

### Edge Encoders (Parallel Structure Pathway)

**Why:** H-map alone is lossy; edge information is important

**Two Variants:**

| Version | Implementation | Output Scales | Use Case |
|---------|---|---|---|
| **v1 (EdgeEncoder)** | Sobel once, then sequential conv | 256, 128, 64, 32 | Lightweight, loses fine details |
| **v2 (MultiScaleEdgeEncoder)** | Recompute Sobel at every scale, feed [h_map, gx, gy] (3ch) to parallel CNNs | 512, 256, 128, 64, 32 | Preserves structure at every resolution (current default) |

**Injection:** Concatenated to decoder skip connections at appropriate resolutions

### Eosin Encoder (Membrane Topology)

**Why:** Hematoxylin H-map only shows nucleus structure; cell membrane boundaries are in Eosin

**Architecture:**
```
Eosin map [B, 1, 512, 512]
→ 5-stage strided conv downsampler
→ [B, 64, 16, 16] at bottleneck
→ Optional: separate projection to [B, 64, 32, 32] at D5
```

**When Enabled:**
- Requires `trainA-E/` and `valA-E/` directories (Eosin channel maps)
- At 16×16 bottleneck: ~30px tissue misalignment becomes <1px — safe to inject
- Optional multi-scale (32×32) is also safe at <2px misalignment

**Disabled if:**
- `--no_use_eosin_encoder` flag passed
- No E-map directories exist (dataset prints warning, returns zeros)

---

## Training Loop & Loss Configuration

### Mixed-Domain Routing (The Key Innovation)

**Problem:** H&E and IHC are consecutive tissue sections, ~30px misaligned. Pixel-level losses penalize something the model can't fix.

**Solution:** Each batch step randomly picks one of two modes:

| Mode | Input | Target | Losses Applied | Frequency |
|------|-------|--------|---|---|
| **Case A (Aligned)** | IHC H-map | IHC RGB | Full-res L1 + LPIPS, DAB block, IHC edge, feat match | 75% (case_b_prob=0.25) |
| **Case B (Misaligned)** | H&E H-map | IHC RGB | Spectral loss, LPIPS @128/256, H&E edge | 25% |
| **Both Cases** | — | — | DAB intensity, DAB histogram, DAB sparsity, adversarial, projection disc | Always |

**Configurable:** `--case_b_prob` (default 0.25). Do NOT go below 0.20 (inference domain shock risk).

### Loss Weights & Configuration (V4 Defaults)

**Tensor Normalization:** All tensors in `[-1, 1]`; UNI tokens in original scale.

**Active Loss Suite (scripts/train/train_mist.py):**

```python
LOSS_WEIGHTS = {
    # === RECONSTRUCTION LOSSES ===
    'l1_fullres_weight': 1.0,        # Case A only: pixel L1 at 512×512
    'lpips_fullres_weight': 1.0,     # Case A only: LPIPS at 512×512
    'lpips_weight': 1.0,             # Case B only: LPIPS @ 128×128
    'lpips_256_weight': 0.5,         # Case B only: LPIPS @ 256×256
    'l1_lowres_weight': 1.0,         # Case B only: FFT spectral loss
    
    # === STRUCTURE/EDGE LOSSES ===
    'he_edge_weight': 0.5,           # Case B: H&E edge preservation
    'ihc_edge_weight': 0.1,          # Case A only: IHC edge supervision
    'edge_weight': 0.0,              # Extra edge sharpness (disabled)
    
    # === DAB/STAIN LOSSES ===
    'dab_intensity_weight': 0.2,     # Both: top-10% DAB mean matching
    'dab_histo_weight': 0.3,         # Both: DAB distribution (W1 distance)
    'dab_block_weight': 0.5,         # Case A only: 16×16 block DAB matching (V4: ↑ from 0.2)
    'dab_sparsity_weight': 0.3,      # Both: hinge on over-staining (V4: new)
    'dab_sparsity_margin': 0.05,     # Margin for sparsity hinge
    'dab_contrast_weight': 0.0,      # Class-ordering (disabled)
    'dab_sharpness_weight': 0.0,     # DAB localization (disabled)
    
    # === ADVERSARIAL LOSSES ===
    'uncond_disc_weight': 1.0,       # Unconditional discriminator (texture judge)
    'proj_disc_weight': 2.0,         # Projection disc (stain-conditioned) (V4: ↑ from 1.0)
    'feat_match_weight': 10.0,       # Feature matching from discriminator
    'adversarial_weight': 0.0,       # Global PatchGAN (disabled)
    'crop_disc_weight': 0.0,         # Crop-level disc (disabled)
    
    # === STYLE/AUX LOSSES ===
    'gram_style_weight': 0.0,        # VGG Gram matrix (disabled)
    'patchnce_weight': 0.0,          # PatchNCE contrastive (disabled)
    'bg_white_weight': 0.0,          # Background whitening (disabled)
}
```

### What Each Loss Does

**Reconstruction (Case A/B Split):**
- `l1_fullres`: Pixel color matching when aligned
- `lpips_fullres`: Perceptual matching at full res (aligned)
- `spectral_loss` (l1_lowres in Case B): FFT magnitude matching (translation-invariant)
- `lpips` at 128/256: Tolerates ~30px drift by using coarse resolution

**Structure:**
- `he_edge`: Sobel edge matching against H&E (Case B, preserves input structure)
- `ihc_edge`: Sobel edge matching against IHC (Case A, aligns with ground truth)

**DAB-Specific (Stain Supervision):**
- `dab_intensity`: Matches top-10% mean intensity (scalar, per-image)
- `dab_histo`: Wasserstein-1 distance on DAB OD distributions (distribution shape)
- `dab_block`: 16×16 block-level DAB mean (spatial placement)
- `dab_sparsity`: Hinge penalty on total DAB mass (prevents over-painting)

**Adversarial:**
- `uncond_disc`: PatchGAN discriminator sees only generated IHC (realism pressure)
- `proj_disc`: Stain-conditioned discriminator (HER2=membrane, Ki67/ER/PR=nuclear)
- `feat_match`: L1 on discriminator intermediate features (texture stability)

### Adversarial Training Timeline

- **Steps 0-2000:** Generator warmup (no adversarial losses)
- **Steps 2000+:** Adversarial losses activated
- **R1 Penalty:** Applied every 16 steps on real images

### Hyperparameter Details

| Param | Default | Notes |
|---|---|---|
| `gen_lr` | 1e-4 | Generator Adam learning rate |
| `disc_lr` | 4e-4 | Discriminator (4× faster) |
| `r1_weight` | 10.0 | R1 gradient penalty strength |
| `r1_every` | 16 | Apply R1 every N steps |
| `adversarial_start_step` | 2000 | Steps before adversarial activation |
| `ema_decay` | 0.999 | EMA generator decay |
| `cfg_drop_class_prob` | 0.10 | Classifier-free guidance: drop class only |
| `cfg_drop_uni_prob` | 0.10 | CFG: drop UNI only |
| `cfg_drop_both_prob` | 0.05 | CFG: drop both class and UNI |

---

## Dataset & Data Flow

### MIST Dataset Layout

**Required Folder Structure:**

```
<data_dir>/
├── HER2/
│   ├── TrainValAB/
│   │   ├── trainA/       # H&E RGB (512×512)
│   │   ├── trainA-H/     # H&E Hematoxylin channel (grayscale)
│   │   ├── trainA-E/     # H&E Eosin channel (optional, grayscale)
│   │   ├── trainB/       # IHC RGB (paired with trainA)
│   │   ├── trainB-H/     # IHC Hematoxylin channel
│   │   ├── valA/
│   │   ├── valA-H/
│   │   ├── valA-E/       # Eosin for validation
│   │   ├── valB/
│   │   └── valB-H/
├── Ki67/  (same structure)
├── ER/    (same structure)
└── PR/    (same structure)
```

**Pairing:** By filename stem (extension-agnostic). E.g., `001.jpg` in `trainA/` pairs with `001.png` in `trainA-H/`.

**When trainA-E missing:** Dataset returns zeros for `he_e_map`, training continues but Eosin encoder receives all-zero input.

### Batch Contract (7-Tuple)

```python
(he_rgb, ihc_rgb, he_h_map, ihc_h_map, he_e_map, stain_label, filename)
```

| Element | Shape | Range | Purpose |
|---------|-------|-------|---------|
| `he_rgb` | [B, 3, H, W] | [-1, 1] | H&E input (for UNI extraction) |
| `ihc_rgb` | [B, 3, H, W] | [-1, 1] | IHC target (for supervised losses) |
| `he_h_map` | [B, 1, H, W] | [-1, 1] | H&E Hematoxylin (generator input in Case B) |
| `ihc_h_map` | [B, 1, H, W] | [-1, 1] | IHC Hematoxylin (generator input in Case A) |
| `he_e_map` | [B, 1, H, W] | [-1, 1] | H&E Eosin (injected at bottleneck/D5) |
| `stain_label` | [B] | {0,1,2,3,4} | 0=HER2, 1=Ki67, 2=ER, 3=PR, 4=null (CFG) |
| `filename` | list[str] | — | For logging & debugging |

**Critical:** All callers (trainer, eval, demo, debug scripts) must handle this 7-tuple consistently.

### Data Augmentations

- **Random crops**: 512×512 (or 1024×1024) from full image
- **Applied identically**: Same random crop to all 4 images (he_rgb, ihc_rgb, he_h_map, ihc_h_map, he_e_map)
- **Spatial augmentations**: Random flips, rotations also applied to all 5 images
- **Normalization**: RGB normalized to [-1,1] with mean=0.5, std=0.5

### UNI Feature Extraction (On-the-Fly)

**Flow (during training):**

1. H&E RGB [B, 3, 512, 512] received in batch
2. Slice into 4×4 grid of 128×128 sub-crops
3. Resize each sub-crop to 224×224
4. Normalize with ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
5. Pass through frozen UNI ViT-L/16 (MahmoodLab/uni)
6. Extract patch tokens: 16×16 = 256 tokens per sub-crop, 1024-dim each
7. Interleave and pool tokens to 32×32 = 1,024 tokens
8. Output: [B, 1024, 1024] token array

**Why on-the-fly:** No need to pre-compute; UNI inference is <1s per batch on A100.

### Stain Label Mapping

```python
STAIN_TO_LABEL = {
    'HER2': 0,
    'Ki67': 1,
    'ER': 2,
    'PR': 3,
}
# null_class = 4 (used for CFG dropout during training)
```

---

## Key Components & File Guide

### src/models/

| File | Class | Purpose |
|------|-------|---------|
| `generator.py` | `SPADEUNetGenerator` | H-map → IHC translation. 42M params. Encoder (1ch), bottleneck, decoder with CrossAttention. |
| `blocks.py` | `CrossAttention`, `SPADE`, `ResBlock`, etc. | Primitive blocks: attention, normalization, residual blocks. |
| `discriminator.py` | `PatchDiscriminator`, `ProjectionDiscriminator`, `MultiScaleDiscriminator` | Discriminators for adversarial training. Projection disc is stain-conditioned. |
| `trainer.py` | `UNIStainNetTrainer` | Lightning module. Orchestrates gen/disc alternation, computes losses, logs metrics, handles EMA. |
| `uni_processor.py` | `UNIFeatureProcessor`, `UNIFeatureProcessorHighRes` | Converts UNI tokens to multi-scale spatial maps (used by SPADE, not primary path). |
| `losses.py` | Various loss functions | LPIPS, Gram, PatchNCE, spectral loss helpers. |
| `edge_encoder.py` | `EdgeEncoder`, `MultiScaleEdgeEncoder`, `EosinEncoder` | Edge extraction (v1/v2) and Eosin bottleneck/multi-scale injection. |

### src/data/

| File | Class | Purpose |
|------|-------|---------|
| `mist_dataset.py` | `MISTMultiStainCropDataset`, `MISTMultiStainCropDataModule` | MIST loader. Multi-stain, random crops, 7-tuple output. |
| `bci_dataset.py` | `BCICropDataset`, `CropPairedDataset` | BCI loader and base class. Class-conditional (HER2 scoring). |

### src/utils/

| File | Purpose |
|------|---------|
| `dab.py` | Ruifrok & Johnston color deconvolution. Extract DAB OD maps. DAB intensity p90 computation. |
| `metrics.py` | All evaluation metrics: FID, KID, LPIPS, SSIM, DAB metrics (intensity, histogram, JSD), IOD, UNI-FID. |

### scripts/train/

| File | Purpose |
|------|---------|
| `train_mist.py` | Multi-stain training entrypoint. Centralizes `LOSS_WEIGHTS` dict. Args for all architectural toggles. |
| `train_mist_1024.py` | Native 1024×1024 training (memory-heavier). |
| `train_bci.py` | BCI HER2 training (class-conditional, fewer stains). |
| `launch_destaining.sh` | One-command launch: unzip MIST → verify folders → CPU sanity checks → train. |

### scripts/eval/

| File | Purpose |
|------|---------|
| `eval_mist.py` | Per-stain evaluation (streaming generation, memory-efficient). Computes all metrics incrementally. |
| `eval_mist_1024.py` | 1024 evaluation (collects outputs in memory). |
| `eval_bci.py` | BCI evaluation with per-class metrics and downstream probes. |

### scripts/debug/

| File | Purpose |
|------|---------|
| `validate_data.py` | CPU data loader check. Prints batch shapes, stain distributions. No model. |
| `sanity_check_mist_dummy.py` | CPU architecture sanity check. Dummy UNI (no download), 1 train + 1 val step. |

### Documentation

| File | Contents |
|------|----------|
| `README.md` | Project intro, installation, usage, architecture summary. |
| `AGENTS.md` | Agent instructions for this repo (conventions, file map, known pitfalls). |
| `changes.md` | Full versioned changelog (V0→V4) with what changed and why. |
| `ABLATION_MATRIX.md` | CLI parameters and ablation study matrix for experiments. |
| `REPORTS/` | Deep architectural explanations and loss history. |
| `reports/` | Codebase explanation and validation vs eval details. |

---

## Practical Workflows

### Workflow 1: Training from Scratch

```bash
cd /path/to/UNISTAINNET

# Step 1: Verify data structure (no GPU needed)
python scripts/debug/validate_data.py \
    --data_dir /path/to/MIST \
    --stains HER2 ER Ki67 PR \
    --batch_size 4 \
    --n_batches 3

# Step 2: CPU architecture sanity check (no GPU, no UNI download)
export PYTHONPATH=$PWD
python scripts/debug/sanity_check_mist_dummy.py

# Step 3: Train (GPU required)
python scripts/train/train_mist.py \
    --data_dir /path/to/MIST \
    --batch_size 8 \
    --max_epochs 100 \
    --wandb_name "my_experiment"

# Alternative: all-in-one on Lightning AI
bash scripts/train/launch_destaining.sh
```

### Workflow 2: Resume from V4 Step-48k (Option A)

```bash
python scripts/train/train_mist.py \
    --data_dir /path/to/MIST \
    --resume_from checkpoints/mist_epoch022_step048000.ckpt \
    --batch_size 8 \
    --wandb_name "resume_v4_option_a"
```

Expects best metrics to stabilize without regression.

### Workflow 3: New Run with Multi-Scale Eosin (Option B)

```bash
python scripts/train/train_mist.py \
    --data_dir /path/to/MIST \
    --eosin_multi_scale \
    --batch_size 8 \
    --max_epochs 100 \
    --wandb_name "option_b_multiscale_eosin"
```

**Important:** `--eosin_multi_scale` changes `EosinEncoder` state_dict keys. Only use for fresh runs.

### Workflow 4: Ablation Study (Disable Edge Encoder)

```bash
python scripts/train/train_mist.py \
    --data_dir /path/to/MIST \
    --edge_encoder none \
    --wandb_name "ablation_no_edge"
```

Other toggles: `--spade_use_uni`, `--use_attention_for_spade`, `--enable_attention_residual`.

### Workflow 5: Evaluation

```bash
python scripts/eval/eval_mist.py \
    --checkpoint checkpoints/mist_multistain/last.ckpt \
    --data_dir /path/to/MIST \
    --stains HER2 Ki67 \
    --batch_size 8
```

Outputs CSV with per-stain metrics: FID, KID, LPIPS, SSIM, DAB p90, DAB KL, etc.

### Workflow 6: Demo / Inference

```python
from src.models.trainer import UNIStainNetTrainer

# Load checkpoint
trainer = UNIStainNetTrainer.load_from_checkpoint('path/to/ckpt.ckpt')

# Single image inference (HF Space / Gradio wrapper, see hf_space/app.py)
he_rgb_tensor = torch.randn(1, 3, 512, 512)  # [-1, 1]
uni_tokens = trainer._extract_uni_from_sub_crops(...)  # [1, 1024, 1024]
stain_label = torch.tensor([0])  # HER2

generated = trainer.generate(he_rgb_tensor, uni_tokens, stain_label, guidance_scale=1.0)
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Folder Name Mismatch

**Symptom:** `FileNotFoundError: 'HER2-Destained' does not exist`

**Cause:** Zip files extract as `HER2-Destained/`, but code expects `HER2/`

**Fix:**
```bash
mv HER2-Destained HER2
mv Ki67-Destained Ki67
mv ER-Destained ER
mv PR-Destained PR
```

Or let `launch_destaining.sh` do it automatically.

### Pitfall 2: HuggingFace Token Missing

**Symptom:** `RuntimeError: HF token not found` or `gated model access denied`

**Cause:** UNI model is gated (MahmoodLab/uni); requires HF token and approval

**Fix:**
```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
huggingface-cli login --token $HF_TOKEN
```

Request access to MahmoodLab/uni on HF Hub first.

### Pitfall 3: Eosin Maps Missing, Generator Crashes

**Symptom:** `shape mismatch` or `expects 1 input, got 0` in eosin_encoder

**Cause:** `trainA-E/` directories don't exist; dataset returns zeros but trainer still tries to inject

**Fix:**
1. Either create the E-map directories (extract Eosin channel from H&E RGB with `src.utils.dab.extract_dab_intensity`)
2. Or disable with `--no_use_eosin_encoder`

### Pitfall 4: VRAM Exceeded

**Symptom:** `CUDA out of memory` at step 1 or during UNI extraction

**Cause:** Batch size too high, or UNI ViT-L/16 processing taking >VRAM

**Fix:**
- Reduce `--batch_size` (8→6 or lower)
- Disable expensive metrics: `--no_log_val_fid --no_log_val_unifid`
- Disable multi-scale discriminators: set `adversarial_weight=0.0, crop_disc_weight=0.0` in LOSS_WEIGHTS

### Pitfall 5: Stain Labels All Look the Same

**Symptom:** Generated HER2 looks like Ki67; no stain-specific differences

**Root Causes:**
1. **Stain embedding not used:** Class embedding computed but discarded (V0 bug from original code)
2. **Cross-Attention too weak:** Attention output negligible; residual makes block nearly identity
3. **UNI tokens too generic:** Pooling compresses out stain-discriminative detail
4. **Losses too forgiving:** Global DAB/distribution losses satisfied without stain-specific structure

**Debug Steps:**
1. Check `class_embed` is actually used in `forward()`: Look for FiLM modulation in SPADE blocks
2. Print attention output magnitudes during training: Should be ~0.1-0.5% of feature magnitude
3. Verify UNI tokens are changing per stain: Print token norms in trainer
4. Try tightening stain-specific losses: Increase `proj_disc_weight`, add `dab_block_weight`

### Pitfall 6: Over-Staining (Ki67/ER/PR Painting Too Many Nuclei)

**Symptom:** Generated image has brown wash everywhere; DAB intensity too high

**Solution (V4 already implemented):**
- Increased `dab_block_weight` to 0.5
- Added `dab_sparsity_weight` to 0.3 with hinge penalty
- Doubled `proj_disc_weight` to 2.0

If still over-staining: increase `dab_sparsity_weight` further (0.5 or 1.0).

### Pitfall 7: Checkpoint Incompatibility with Multi-Scale Eosin

**Symptom:** `KeyError: eosin_encoder.stage1.0.weight` when loading old checkpoint with `--eosin_multi_scale`

**Cause:** Option B (`--eosin_multi_scale`) changes EosinEncoder state_dict keys from sequential to named stages

**Fix:** Only use `--eosin_multi_scale` for fresh runs. Do NOT mix with `--resume_from`.

---

## Evaluation & Metrics

### Image Quality Metrics

| Metric | Computed How | Pair | Better If |
|--------|---|---|---|
| **FID** | Inception features, Fréchet distance | Generated vs Real IHC | Lower (< 35 is good) |
| **KID** | Kernel Inception Distance (MMD) | Generated vs Real IHC | Lower |
| **LPIPS** | Alex CNN perceptual features | Generated vs Real IHC | Lower |
| **SSIM** | Structural Similarity Index | Generated vs Real IHC | Higher (0.0-1.0) |
| **PSNR** | Peak Signal-to-Noise Ratio | Generated vs Real IHC | Higher |

### DAB-Specific Metrics

| Metric | What It Measures | Computed How | Better If |
|---|---|---|---|
| **DAB p90** | Top-10% mean DAB intensity | Deconvolved OD, top-10% pixels | Matches real image p90 |
| **DAB MAE** | Absolute error in p90 | Mean \|gen_p90 - real_p90\| | Lower (< 0.05 is excellent) |
| **DAB KL** | Divergence in DAB distribution | KL(gen_hist \|\| real_hist) | Lower |
| **DAB JSD** | Jensen-Shannon divergence | JS(gen_hist, real_hist) | Lower |
| **DAB Pearson-r** | Correlation in p90 scores | Pearson correlation across images | Higher (> 0.9 is excellent) |

### Structure Metrics

| Metric | What It Measures | Computed How | Better If |
|---|---|---|---|
| **H-Structure SSIM** | Edge preservation from H&E | Sobel edge map SSIM between gen IHC and H&E | Higher |
| **H-Channel SSIM** | H-map reconstruction | SSIM of H-channel extracted from generated | Higher |

### Validation vs Evaluation

**Validation (during training, every epoch):**
- Random 512×512 crops from val split
- LPIPS, SSIM, DAB MAE computed per batch
- FID/KID/UNI-FID streamed (optional, slower)
- Logged to W&B

**Evaluation (standalone scripts):**
- Entire val/test split (or random crops if specified)
- All metrics computed incrementally (streaming in eval_mist.py)
- Output to CSV per stain
- No real-time plotting; post-hoc analysis

**Why results differ:** Validation uses random crops; evaluation may use full or different crops. Metrics are slightly lower during eval (larger sample size, less overfitting to val set).

---

## Development Guidelines

### Code Style & Conventions

1. **Tensor Ranges:** Always document input/output ranges
   - Most I/O: `[-1, 1]` (normalized with mean=0.5, std=0.5)
   - UNI tokens: original scale (ImageNet normalized)

2. **Shapes:** Document [B, C, H, W] notation clearly
   ```python
   x = model(input)  # input: [B, 1, 512, 512], output: [B, 3, 512, 512]
   ```

3. **Comments:** For non-obvious design decisions, explain the "why"
   ```python
   # Case A only: real IHC and H-map are aligned, so pixel-level loss is safe
   if use_aligned:
       loss_l1 = l1_loss(generated, target)
   ```

4. **File Organization:** Keep each module focused
   - `generator.py`: only the main generator
   - `blocks.py`: reusable blocks (attention, SPADE, etc.)
   - `losses.py`: all loss functions
   - `trainer.py`: orchestration (not model definition)

### Adding New Losses

1. **Implement in `src/models/losses.py`** or inline in `trainer.py`:
   ```python
   def compute_my_loss(generated, target, mask=None):
       """Custom loss description.
       
       Args:
           generated: [B, 3, H, W] in [-1, 1]
           target: [B, 3, H, W] in [-1, 1]
       Returns:
           scalar loss
       """
       ...
       return loss
   ```

2. **Add weight param to LOSS_WEIGHTS dict** in `train_mist.py`:
   ```python
   'my_loss_weight': 0.1,
   ```

3. **Compute & log in trainer.training_step():**
   ```python
   my_loss = compute_my_loss(generated, target) * my_loss_weight
   total_loss += my_loss
   self.log('train/my_loss', my_loss, ...)
   ```

4. **Test with ablation:**
   ```bash
   # Disable: set weight to 0.0
   # Or set to 0.0 before running
   ```

### Adding New Metrics

1. **Implement in `src/utils/metrics.py`:**
   ```python
   def compute_my_metric(generated, target):
       """Compute my metric.
       
       Args:
           generated: [B, 3, H, W] or [N, 3, H, W]
           target: same shape
       Returns:
           dict: {'my_metric': scalar, ...}
       """
       ...
       return {'my_metric': value}
   ```

2. **Add to eval script** (eval_mist.py / eval_bci.py):
   ```python
   metrics = compute_my_metric(generated, target)
   all_metrics['my_metric'] = metrics['my_metric']
   ```

3. **Optional: add to trainer validation_step() for logging during training**

### Testing New Changes

1. **CPU-only (no GPU):**
   ```bash
   python scripts/debug/validate_data.py --data_dir /path/to/MIST
   python scripts/debug/sanity_check_mist_dummy.py
   ```

2. **Single GPU with small batch:**
   ```bash
   python scripts/train/train_mist.py \
       --data_dir /path/to/MIST \
       --batch_size 2 \
       --max_epochs 1 \  # Just 1 epoch for testing
       --wandb_name "test_change"
   ```

3. **Evaluate a small subset:**
   ```bash
   python scripts/eval/eval_mist.py \
       --checkpoint ckpt.ckpt \
       --data_dir /path/to/MIST \
       --stains HER2 \  # Just 1 stain
       --batch_size 4
   ```

### When Modifying Architecture

**Backward Compatibility:**
- Always provide CLI flags for new features (e.g., `--eosin_multi_scale`)
- Make new flags default to current behavior (don't break existing scripts)
- Document state_dict incompatibilities clearly (e.g., Option B Eosin)

**Update Multiple Places:**
- Generator definition (`src/models/generator.py`)
- Trainer instantiation (`src/models/trainer.py`, `scripts/train/*.py`)
- Forward signature (if new inputs)
- Batch unpacking in trainer
- Eval scripts (if they call generator.forward directly)
- Demo/HF space app (if applicable)

**Test with checkpoint loading:**
```python
trainer = UNIStainNetTrainer.load_from_checkpoint('old_ckpt.ckpt')
# Should not crash; should use old architecture params
```

---

## Summary: Current State & Next Steps

### What We Have (V4, May 2026)

1. ✅ **Structure-semantic decoupling** works (H-map encoder + UNI tokens)
2. ✅ **Mixed-domain training** handles misalignment robustly (75/25 routing)
3. ✅ **Eosin injection** at bottleneck improves membrane topology
4. ✅ **Stain-specific supervision** via projection discriminator + DAB losses
5. ✅ **Comprehensive ablations** (edge encoder, attention, SPADE variants)

### Current Issues (Observed at Epoch 24)

1. 🔴 **HER2 nuclear shortcut:** Membrane localization still weak (proj_adv_g oscillating)
2. 🔴 **Ki67/ER/PR over-staining:** Nuclei painted too much (though V4 helps)
3. 🔴 **Metric regression at step 52k:** Val/SSIM drops, val/DAB MAE regresses

### Two Options Forward

| Option | Approach | Status | Try This |
|---|---|---|---|
| **A (Resume)** | Resume step-48k with V4 tightened weights | Ready | `--resume_from checkpoints/.../step048000.ckpt --batch_size 8 --max_epochs 150` |
| **B (Fresh)** | New run with multi-scale Eosin (32×32 injection at D5) | Experimental | `--eosin_multi_scale --batch_size 8 --max_epochs 100` |

### Questions to Explore Next

1. **Why does HER2 proj_disc still oscillate?** Is the discriminator too unstable? Try reducing learning rate or increasing R1 penalty.
2. **Can Eosin multi-scale (Option B) resolve the nuclear shortcut?** Finer membrane signal at 32×32 might help.
3. **Is the stain embedding actually used?** Verify SPADE FiLM modulation is working (compute gradients w.r.t. class_embed).
4. **Should we try a different stain-conditioning approach?** E.g., class-conditioned attention or separate nuclei/membrane pathways.

---

## Reference: Key Paper & Code Links

- **Paper:** https://arxiv.org/abs/2603.12716
- **Project Page:** https://facevoid.github.io/UNIStainNet/
- **Demo:** https://huggingface.co/spaces/faceless-void/UNIStainNet
- **UNI Model:** https://github.com/mahmoodlab/UNI

---

**End of system_explain.md**

For detailed information on any section, refer to:
- Architecture details → `REPORTS/UNIStainNet_ARCHITECTURE_AND_LOSSES.md`
- Training history → `REPORTS/losses\ through\ history.md`
- Loss & metric explanations → `REPORTS/evaluation\ metrics.md`
- Version changelog → `changes.md`
