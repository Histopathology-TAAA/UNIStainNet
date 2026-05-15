# UNIStainNet Ablation Study & Parameter Reference

## Edge Encoder: v1 vs v2

### v1 (`EdgeEncoder`)
- **Mechanism**: Computes Sobel gx/gy once on the 512×512 H-map input, then runs those 2 channels through a sequential downsampling encoder.
- **Output scales**: 256, 128, 64, 32 (no 512-resolution output).
- **Use case**: Lightweight edge pathway; loses fine details during sequential pooling.
- **Code**: [src/models/edge_encoder.py](src/models/edge_encoder.py#L21)

### v2 (`MultiScaleEdgeEncoder`)
- **Mechanism**: Recomputes Sobel independently at every scale (512, 256, 128, 64, 32) and feeds `[h_map, gx, gy]` (3 channels) into separate CNNs at each resolution.
- **Output scales**: 512, 256, 128, 64, 32 (includes 512-resolution edge features).
- **Advantages**: Raw H-map preserved at each scale; stronger fine-structure signal; scale-specific edge extraction.
- **Use case**: Stronger structure preservation; current default in training script.
- **Code**: [src/models/edge_encoder.py](src/models/edge_encoder.py#L86)

**Practical difference**: v1 = one Sobel pass + hierarchy. v2 = multi-resolution Sobel + H-map preservation at every scale.

---

## CLI Parameters (Runtime Options)

All parameters below are exposed through `scripts/train/train_mist.py` as command-line flags.

| Parameter | Base Value | Type | Description |
|---|---:|---|---|
| `--edge_encoder` | `v2` | choice: `none`, `v1`, `v2` | Selects edge pathway. `none` disables all edge features. `v1` uses single-pass Sobel. `v2` uses multi-scale Sobel (default). |
| `--use_attention_for_spade` | `False` | flag | If enabled, SPADE blocks receive decoder cross-attention maps instead of raw UNI spatial maps as spatial conditioning. |
| `--enable_attention_residual` | `True` | flag | If enabled, each decoder stage adds an independent attention residual update `x = x + (x_attn - x_conv)`. Disable for simpler decoder. |
| `--spade_use_uni` | `True` | flag | If False, SPADE ignores UNI spatial maps and becomes stain-label-only (FiLM modulation only). Enables "no UNI in SPADE" ablation. |
| `--use_eosin_encoder` | `True` | flag | Enables Eosin bottleneck injection. Requires `trainA-E/` data directories. |
| `--eosin_multi_scale` | `False` | flag | Adds second Eosin injection at D5 (32×32). Incompatible with existing checkpoints — use only for fresh runs. |
| `--case_b_prob` | `0.25` | float | Mixed-domain routing: probability of Case B (misaligned H&E-H input). 0.25 = 75% aligned, 25% misaligned. Do NOT go below 0.20. |
| `--resume_from` | `None` | path | Checkpoint path for resuming training. Passed to `trainer.fit(..., ckpt_path=...)`. |
| `--wandb_name` | `mist_multistain` | string | W&B run name. |
| `--batch_size` | `16` | int | Data loader batch size. Recommend 8–12 on A100 40GB; 4–6 on A10G. |
| `--max_epochs` | `100` | int | Maximum training epochs. |
| `--stains` | `HER2 Ki67 ER PR` | list | Stain folders to include in training. |
| `--ckpt_dir` | `checkpoints/mist_multistain` | path | Local checkpoint output directory. |
| `--data_dir` | *required* | path | Path to MIST dataset root (contains `HER2/`, `Ki67/`, etc.). |

---

## Fixed Architecture Parameters (Not CLI-Exposed)

These are currently hardcoded in `scripts/train/train_mist.py` but can be edited in the script if needed.

| Parameter | Base Value | Description |
|---|---:|---|
| `class_dim` | `64` | Stain embedding dimension (input to FiLM in SPADE). |
| `uni_dim` | `1024` | UNI token embedding width from ViT-L/16. |
| `input_skip` | `True` | Concatenates H-map input skip at output layer for residual connection. |
| `edge_base_ch` | `32` | Base channels for edge encoder feature maps. |
| `uni_spatial_size` | `32` | UNI spatial map size (32×32 patch tokens for decoder conditioning). |
| `extract_uni_on_the_fly` | `True` | Extracts UNI features online from H&E RGB during training (not pre-computed). |
| `uni_spatial_pool_size` | `32` | Pool size for UNI spatial token grid (usually matches `uni_spatial_size`). |
| `gen_lr` | `1e-4` | Generator Adam learning rate. |
| `disc_lr` | `4e-4` | Discriminator Adam learning rate. |
| `r1_weight` | `10.0` | R1 gradient penalty strength. |
| `r1_every` | `16` | Apply R1 penalty every N steps. |
| `adversarial_start_step` | `2000` | Delay (steps) before adversarial training starts (generator warm-up). |
| `ema_decay` | `0.999` | Exponential moving average decay for EMA generator. |
| `cfg_drop_class_prob` | `0.10` | Classifier-free guidance dropout probability for class conditioning. |
| `cfg_drop_uni_prob` | `0.10` | CFG dropout probability for UNI conditioning. |
| `cfg_drop_both_prob` | `0.05` | CFG dropout probability for both class and UNI. |

---

## Loss Weights (Base V4 Tuned Defaults)

These are defined in `LOSS_WEIGHTS` dictionary in `scripts/train/train_mist.py`. Edit the dictionary to adjust weights for your ablation studies.

### Core Reconstruction Losses

| Loss Weight | Base Value | Cases | Description |
|---|---:|---|---|
| `l1_fullres_weight` | `1.0` | A only | Pixel-wise L1 loss at full resolution (aligned pairs only). |
| `lpips_fullres_weight` | `1.0` | A only | LPIPS perceptual loss at full resolution (aligned pairs only). |
| `lpips_weight` | `1.0` | B only | LPIPS at 128×128 (misaligned pairs; tolerates ~30px drift). |
| `lpips_256_weight` | `0.5` | B only | LPIPS at 256×256 (misaligned; finer tolerance). |
| `lpips_512_weight` | `0.0` | disabled | LPIPS at 512×512 (not used in current config). |
| `l1_lowres_weight` | `1.0` | B only | Spectral/FFT loss for misaligned boundary sharpness. |

### Structure/Edge Losses

| Loss Weight | Base Value | Cases | Description |
|---|---:|---|---|
| `ihc_edge_weight` | `0.1` | A only | Sobel edge loss on IHC target (pixel-aligned boundary supervision). |
| `he_edge_weight` | `0.5` | B only | Sobel edge loss on H&E input (structure preservation in misaligned case). |
| `edge_weight` | `0.0` | disabled | FFT-based spectral boundary sharpness loss (disabled). |
| `dab_block_size` | `32` | — | Block size for DAB block-level losses (32 → 16×16 blocks at 512×512). |

### DAB/Stain-Specific Losses

| Loss Weight | Base Value | Cases | Description |
|---|---:|---|---|
| `dab_intensity_weight` | `0.2` | Both | Top-10% DAB intensity matching between generated and real. |
| `dab_histo_weight` | `0.3` | Both | Wasserstein-1 OD histogram distribution matching. |
| `dab_block_weight` | `0.5` | A only | Block-level (16×16) DAB mean spatial placement matching. |
| `dab_sparsity_weight` | `0.3` | Both | Hinge penalty for over-staining (generated DAB > real + margin). |
| `dab_sparsity_margin` | `0.05` | — | Margin threshold for sparsity hinge loss. |
| `dab_contrast_weight` | `0.0` | disabled | Class-ordering DAB hinge (HER2 > Ki67/ER/PR; currently disabled). |
| `dab_sharpness_weight` | `0.0` | disabled | DAB membrane localization sharpness loss (currently disabled). |

### Adversarial Losses

| Loss Weight | Base Value | Description |
|---|---:|---|
| `adversarial_weight` | `0.0` | Global multi-scale PatchGAN adversarial loss (currently disabled). |
| `uncond_disc_weight` | `1.0` | Unconditional discriminator (alignment-free texture judge). Primary adversarial path. |
| `proj_disc_weight` | `2.0` | Stain-conditioned projection discriminator (stain-specific realism). |
| `feat_match_weight` | `10.0` | Feature matching loss from unconditional discriminator intermediate features. |
| `crop_disc_weight` | `0.0` | Crop discriminator on random patches (currently disabled). |

### Style/Auxiliary Losses

| Loss Weight | Base Value | Description |
|---|---:|---|
| `gram_style_weight` | `0.0` | VGG Gram-matrix style loss (currently disabled). |
| `patchnce_weight` | `0.0` | PatchNCE contrastive loss (alignment-free; currently disabled). |
| `bg_white_weight` | `0.0` | Background white penalty loss (currently disabled). |

---

## Ablation Study Matrix

### Setup 1: Baseline (Full UNI in All Paths)
```bash
python scripts/train/train_mist.py \
  --data_dir /path/to/MIST \
  --edge_encoder v2 \
  --no_use_attention_for_spade \
  --enable_attention_residual \
  --spade_use_uni \
  --use_eosin_encoder \
  --wandb_name "baseline_v2_full_uni"
```
**What**: Full model with v2 edges, UNI in SPADE, UNI in attention, Eosin bottleneck.

---

### Setup 2: Attention-Only UNI in Decoder + Label-Only SPADE
```bash
python scripts/train/train_mist.py \
  --data_dir /path/to/MIST \
  --edge_encoder none \
  --use_attention_for_spade \
  --enable_attention_residual \
  --no_spade_use_uni \
  --use_eosin_encoder \
  --wandb_name "attention_only_label_spade"
```
**What**: UNI only in cross-attention path; SPADE becomes stain-label-only (FiLM only); no edge pathway.

---

### Setup 3: No Edge Encoder (Baseline Structure from Encoder Only)
```bash
python scripts/train/train_mist.py \
  --data_dir /path/to/MIST \
  --edge_encoder none \
  --no_use_attention_for_spade \
  --enable_attention_residual \
  --spade_use_uni \
  --use_eosin_encoder \
  --wandb_name "no_edge_encoder"
```
**What**: No explicit edge pathway; structure comes only from encoder skip connections + SPADE/FiLM.

---

### Setup 4: No Attention Residual (SPADE + Conv Only)
```bash
python scripts/train/train_mist.py \
  --data_dir /path/to/MIST \
  --edge_encoder v2 \
  --no_use_attention_for_spade \
  --disable_attention_residual \
  --spade_use_uni \
  --use_eosin_encoder \
  --wandb_name "no_attention_residual"
```
**What**: Removes per-stage attention residual; decoder uses only conv + SPADE + FiLM.

---

### Setup 5: v1 Edge Encoder (Single-Pass Sobel)
```bash
python scripts/train/train_mist.py \
  --data_dir /path/to/MIST \
  --edge_encoder v1 \
  --no_use_attention_for_spade \
  --enable_attention_residual \
  --spade_use_uni \
  --use_eosin_encoder \
  --wandb_name "v1_edge_encoder"
```
**What**: v1 single-pass Sobel edge encoder instead of v2 multi-scale.

---

### Setup 6: Maximum Structure + No UNI in SPADE
```bash
python scripts/train/train_mist.py \
  --data_dir /path/to/MIST \
  --edge_encoder v2 \
  --use_attention_for_spade \
  --enable_attention_residual \
  --no_spade_use_uni \
  --use_eosin_encoder \
  --eosin_multi_scale \
  --wandb_name "max_structure_no_spade_uni"
```
**What**: v2 edges + multi-scale Eosin + attention conditioning + attention residual + label-only SPADE.

---

### Setup 7: Minimal (Label-Only, No Structure Pathways)
```bash
python scripts/train/train_mist.py \
  --data_dir /path/to/MIST \
  --edge_encoder none \
  --no_use_attention_for_spade \
  --disable_attention_residual \
  --no_spade_use_uni \
  --no_use_eosin_encoder \
  --wandb_name "minimal_label_only"
```
**What**: Encoder-only structure; SPADE is pure stain embedding (FiLM); no attention, no edges, no Eosin.

---

## Checkpoint Compatibility Notes

### **Safe Resume Cases** (same or compatible architectures)
- Changing only loss weights: ✅ Resume with same checkpoint.
- Changing only `case_b_prob` or CFG dropout: ✅ Resume with same checkpoint.
- Disabling attention residual: ✅ Resume (attention residual path is optional).
- Changing `--use_attention_for_spade`: ✅ Resume (SPADE conditioning is internal).

### **Requires Fresh Checkpoint** (architecture incompatibility)
- Changing `--edge_encoder` to `none` or different mode: ❌ Fresh run.
- Enabling `--eosin_multi_scale` when checkpoint has it disabled: ❌ Fresh run (state_dict keys change).
- Enabling `--eosin_multi_scale` from a checkpoint without it: ❌ Fresh run.
- Changing `--spade_use_uni` from True to False (or vice versa): ⚠️ May work but not recommended; fresh run safer.

---

## W&B Logging

All runs automatically log:

- **Hyperparameters**: All CLI args + fixed architecture params via `save_hyperparameters()`.
- **Scalar metrics**: Every loss term via `self.log('train/loss_name', value)` at every step.
- **Validation metrics**: `val/lpips`, `val/ssim`, `val/dab_mae` per epoch.
- **Learning rate**: Tracked by `LearningRateMonitor`.
- **Checkpoints**: Saved locally in `--ckpt_dir` and logged as artifacts (if W&B artifact logging is enabled).

Project name: `Destaining-UNIStainNet-V1` (hardcoded in script; modify `wandb_name` to change run name).

---

## Loss Weight Tuning Guide

**For a new run**, do a short calibration pass:

1. Run 1 epoch with all weights at 1.0 to see raw loss scales.
2. Collect the mean value of each active loss.
3. Scale each weight so that the weighted losses end up in a similar magnitude band (e.g., 0.1–1.0).

**Example calibration approach**:
```python
# From W&B logs after 1 epoch:
# train/l1_fullres: mean ~0.15
# train/lpips_fullres: mean ~0.8
# train/dab_intensity: mean ~0.05
# train/feat_match: mean ~2.0

# Rescale to 0.1–1.0 range:
l1_fullres_weight = 0.5       # 0.15 * 0.5 = 0.075
lpips_fullres_weight = 1.0    # 0.8 * 1.0 = 0.8 (no change)
dab_intensity_weight = 2.0    # 0.05 * 2.0 = 0.1
feat_match_weight = 0.5       # 2.0 * 0.5 = 1.0
```

Then re-run with the rescaled weights.

---

## Quick Start Commands

**Install & verify**:
```bash
cd ~/teamspace/studios/this_studio/UNISTAINNET
pip install -r requirements.txt -q && pip install -e . -q
export HF_TOKEN=hf_xxxx  # Your Hugging Face token for UNI model access
```

**CPU sanity checks** (free, no GPU):
```bash
python scripts/debug/sanity_check_mist_dummy.py
python scripts/debug/validate_data.py --data_dir ~/teamspace/studios/this_studio/data/MIST --stains HER2 ER Ki67 PR --batch_size 4 --n_batches 3
```

**Run ablation study (6 setups)**:
```bash
# Setup 1
python scripts/train/train_mist.py --data_dir /path/to/MIST --edge_encoder v2 --wandb_name "baseline_v2_full_uni" --batch_size 8

# Setup 2
python scripts/train/train_mist.py --data_dir /path/to/MIST --edge_encoder none --use_attention_for_spade --no_spade_use_uni --wandb_name "attention_only_label_spade" --batch_size 8

# ... (repeat for remaining setups)
```

**Resume from checkpoint**:
```bash
python scripts/train/train_mist.py \
  --data_dir /path/to/MIST \
  --edge_encoder v2 \
  --resume_from checkpoints/mist_multistain/mist_epoch_005_step_10000.ckpt \
  --max_epochs 150 \
  --wandb_name "baseline_v2_full_uni_resumed"
```

---

## Key References in Codebase

- **Ablation switches**: [scripts/train/train_mist.py](scripts/train/train_mist.py#L76)
- **Generator architecture**: [src/models/generator.py](src/models/generator.py#L30)
- **SPADE block (label-only ablation)**: [src/models/blocks.py](src/models/blocks.py#L30)
- **Edge encoders (v1 vs v2)**: [src/models/edge_encoder.py](src/models/edge_encoder.py#L21)
- **Trainer & losses**: [src/models/trainer.py](src/models/trainer.py#L110)
- **W&B logger config**: [scripts/train/train_mist.py](scripts/train/train_mist.py#L198)
