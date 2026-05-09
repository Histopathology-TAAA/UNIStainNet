# AGENTS.md

Agent instructions for this repository.

## Scope
- Applies to the entire repository.
- Keep changes focused and minimal. Do not refactor unrelated modules.
- Active branch: `exp-destaining-v1-attention` — destaining variant is the main work.
- **Current version: V3** (see [changes.md](changes.md) for full versioned changelog).

---

## First Read
- Project overview, setup, dataset layout, and baseline commands: [README.md](README.md)
- Full versioned changelog (V0 → V3): [changes.md](changes.md)
- Demo app entrypoint (Gradio/HF Space): [hf_space/app.py](hf_space/app.py)

---

## What This Code Does (Destaining Approach)

UNIStainNet translates H&E (Hematoxylin & Eosin) histopathology images into IHC
(Immunohistochemistry) stained images. The `exp-destaining-v1-attention` branch
uses a structure-semantic decoupled approach:

1. **Destain**: Extract the Hematoxylin channel (H-map, 1-ch grayscale) from H&E — pure tissue structure, no stain color.
2. **Encode structure**: UNet encoder processes only the H-map.
3. **Inject semantics**: UNI ViT-L/16 (frozen) extracts 32×32 patch tokens from H&E RGB. These are injected at every decoder level via **Cross-Attention** (Q = spatial features, K/V = UNI tokens).
4. **Restain**: Decoder synthesizes the target IHC stain guided by UNI semantics + optional edge encoder.

The generator never sees raw RGB at training-time — only the H-map goes in. UNI tokens carry the biological meaning.

---

## Architecture Map

### Core Model
| File | Role |
|------|------|
| [src/models/generator.py](src/models/generator.py) | SPADEUNetGenerator — 1ch H-map encoder + CrossAttention decoder + Eosin injection |
| [src/models/blocks.py](src/models/blocks.py) | CrossAttention, SelfAttention, ResBlock, SPADEBlock |
| [src/models/edge_encoder.py](src/models/edge_encoder.py) | EdgeEncoder (v1) / MultiScaleEdgeEncoder (v2) — parallel structure pathway; **EosinEncoder** — 512→16 bottleneck compressor |
| [src/models/uni_processor.py](src/models/uni_processor.py) | UNIFeatureProcessor / UNIFeatureProcessorHighRes — UNI token pipeline |
| [src/models/discriminator.py](src/models/discriminator.py) | PatchGAN + MultiScaleDiscriminator + **ProjectionDiscriminator** (stain-conditioned) + loss functions |
| [src/models/trainer.py](src/models/trainer.py) | UNIStainNetTrainer — Lightning GAN loop, all losses, CFG dropout, EMA |
| [src/models/losses.py](src/models/losses.py) | VGG/Gram, PatchNCE |
| [src/utils/dab.py](src/utils/dab.py) | DAB deconvolution for intensity/contrast losses |
| [src/utils/metrics.py](src/utils/metrics.py) | LPIPS, SSIM, DAB MAE |

### Data
| File | Role |
|------|------|
| [src/data/mist_dataset.py](src/data/mist_dataset.py) | MISTMultiStainCropDataset / DataModule — main dataset for destaining |
| [src/data/bci_dataset.py](src/data/bci_dataset.py) | CropPairedDataset base class + BCI/MIST single-stain variants |

### Scripts
| File | Role |
|------|------|
| [scripts/train/train_mist.py](scripts/train/train_mist.py) | Multi-stain training entrypoint |
| [scripts/train/launch_destaining.sh](scripts/train/launch_destaining.sh) | **Full Lightning AI launch script** — unzip, verify, install, sanity check, train |
| [scripts/debug/sanity_check_mist_dummy.py](scripts/debug/sanity_check_mist_dummy.py) | CPU dummy-data architecture check — no GPU, no UNI download |
| [scripts/debug/validate_data.py](scripts/debug/validate_data.py) | CPU real-data loading check — no model, just verifies folders + shapes |

---

## Training Logic (Key Design Decisions)

### Mixed-Domain Routing (configurable, default 75/25)
Each batch randomly picks one of two modes. Split is controlled by `case_b_prob`
(probability of Case B; default `0.25` — 75% aligned, 25% misaligned).
Do NOT go below `case_b_prob=0.20` or the H&E H-map domain becomes OOD at inference.

| Mode | Generator Input | Losses Applied |
|------|----------------|----------------|
| **Aligned (Case A)** — 75% | IHC H-map | Full-res L1 + full-res LPIPS + IHC Sobel edge + DAB block |
| **Misaligned (Case B)** — 25% | H&E H-map | FFT spectral loss + LPIPS @ 128 & 256 + H&E Sobel edge |
| **Both cases** | — | DAB intensity + DAB histogram (W1) + proj_disc adversarial + uncond_disc adversarial |

UNI tokens are **always extracted from H&E RGB** regardless of mode.
Eosin maps (`he_e_map`) are **always from H&E**, injected at the bottleneck in both cases.
Validation **always** uses H&E H-map → IHC RGB (inference-time domain).

### CFG Dropout (trainer.py:331)
| Dropped | Probability |
|---------|-------------|
| Class only | 10% |
| UNI only | 10% |
| Both | 5% |

### EMA
Generator EMA (decay=0.999) is used for all validation and inference — never the raw generator.

### Adversarial Warmup
Discriminator losses are activated only after `adversarial_start_step=2000` steps.

---

## Active Losses (V3 defaults in train_mist.py)

| W&B key | Weight | Cases | What it penalizes |
|---------|--------|-------|-------------------|
| `train/l1_fullres` | 1.0 | A only | Pixel color error (full-res, aligned) |
| `train/lpips_fullres` | 1.0 | A only | Perceptual error (full-res, aligned) |
| `train/spectral_misalign` | 1.0 | B only | Wrong frequency content (FFT, translation-invariant) |
| `train/lpips` | 1.0 | B only | Perceptual error @ 128px (tolerates 30px drift) |
| `train/lpips_fine` | 0.5 | B only | Perceptual error @ 256px |
| `train/ihc_edge` | 0.1 | A only | Wrong membrane/boundary structure vs real IHC Sobel |
| `train/he_edge` | 0.5 | B only | Missing H&E nuclei structure in generated IHC |
| `train/dab_intensity` | 0.2 | Both | Wrong top-10% DAB mean intensity |
| `train/dab_histo` | 0.3 | Both | Wrong OD distribution shape (Wasserstein-1) |
| `train/dab_block` | 0.2 | A only | Wrong spatial DAB placement (16×16 blocks) |
| `train/feat_match` | 10.0 | A only | Texture statistics mismatch (disc intermediate features) |
| `train/uncond_adv_g` | 1.0 | Both | Unconditional realism (after step 2000) |
| `train/proj_adv_g` | 1.0 | Both | Stain-specific realism — HER2 membrane, Ki67/ER/PR nuclear (after step 2000) |

**To disable any loss:** set its weight to `0.0` in `scripts/train/train_mist.py`. Most losses do
not instantiate any module when set to 0 (the exception is `proj_disc_weight` which also
controls discriminator instantiation — setting it to 0 prevents the discriminator from
being created at all).

---

## Dataset Layout (Destaining)

### Required Folder Structure

```
<data_dir>/
  HER2/
    trainA/       ← H&E RGB images (paired)
    trainA-H/     ← H&E Hematoxylin channel (grayscale, same stems as trainA)
    trainA-E/     ← H&E Eosin channel (grayscale) — loaded if present; zeros if absent
    trainB/       ← IHC RGB images (paired with trainA)
    trainB-H/     ← IHC Hematoxylin channel (grayscale, same stems as trainB)
    valA/
    valA-H/
    valA-E/       ← Eosin channel for validation
    valB/
    valB-H/
  ER/
    (same structure)
  Ki67/
    (same structure)
  PR/
    (same structure)
```

**Pairing is by filename stem (extension-agnostic).**
A file `001.jpg` in `trainA/` pairs with `001.png` in `trainA-H/` as long as stems match.
When `trainA-E/` exists, only stems present in all 5 directories (A, A-H, A-E, B, B-H) are used.

### Stain Label Mapping (STAIN_TO_LABEL in mist_dataset.py)
| Folder name | Label |
|-------------|-------|
| HER2 | 0 |
| Ki67 | 1 |
| ER | 2 |
| PR | 3 |
| null (CFG) | 4 |

**Critical**: The downloaded zips extract as `HER2-Destained/`, `ER-Destained/`, etc.
These must be renamed to `HER2/`, `ER/`, `Ki67/`, `PR/` — the launch script handles this automatically.

### Batch Tuple (7-element, as of V2)
`(he_rgb, ihc_rgb, he_h_map, ihc_h_map, he_e_map, stain_label, filename)`
- `he_e_map`: `[B, 1, 512, 512]` in `[-1, 1]`. Zeros if `trainA-E/` does not exist.

### Extra Folders (Not Loaded)
`trainB-DAB`, `valB-DAB` — present in zips but not used.

---

## Lightning AI Setup

### File Layout Assumption
```
/teamspace/studios/this_studio/   ← this IS $HOME on Lightning AI, do NOT use ~/teamspace/...
    UNISTAINNET/          ← this repo
    Destained_Results/    ← downloaded zips: HER2-Destained.zip, ER-Destained.zip, ...
    data/MIST/            ← created by launch script
    checkpoints/          ← created by training
```

### Pre-GPU Validation (CPU-only, free)

Run these in order BEFORE enabling a GPU. Steps 1-3 cost nothing.

**Step 1 — Import check:**
```bash
cd ~/teamspace/studios/this_studio/UNISTAINNET
pip install -r requirements.txt -q && pip install -e . -q
python -c "import torch, timm, lpips, pytorch_lightning, torchmetrics; print('All imports OK')"
```

**Step 2 — Architecture sanity check (dummy data, no UNI download):**
```bash
export PYTHONPATH=$PWD
python scripts/debug/sanity_check_mist_dummy.py
# Expected: prints batch shapes, runs 1 train + 1 val step, prints "Sanity check complete."
```

**Step 3 — Real data loading check (no model, no GPU):**
```bash
python scripts/debug/validate_data.py \
    --data_dir ~/teamspace/studios/this_studio/data/MIST \
    --stains HER2 ER Ki67 PR \
    --batch_size 4 \
    --n_batches 3
# Expected: prints sample shapes in [-1, 1], stain distributions, "All checks passed."
```
If validate_data.py fails with `FileNotFoundError`, the folder name is wrong — re-check unzip output with `ls data/MIST/`.

---

## Training Launch

### One-Command Launch (unzip + verify + install + train)
```bash
cd ~/teamspace/studios/this_studio/UNISTAINNET
bash scripts/train/launch_destaining.sh
```
The script: unzips → renames folders → verifies 8 subfolders per stain → installs deps → CPU sanity check → CPU data check → starts GPU training.

### Manual Training Command
```bash
cd /teamspace/studios/this_studio/UNISTAINNET
export PYTHONPATH=$PWD
python scripts/train/train_mist.py \
    --data_dir   /teamspace/studios/this_studio/data/MIST \
    --stains     HER2 ER Ki67 PR \
    --batch_size 8 \
    --max_epochs 100 \
    --ckpt_dir   /teamspace/studios/this_studio/checkpoints/destaining_v1 \
    --wandb_name destaining_v1_attention_batch8

# To disable Eosin injection (if trainA-E dirs are absent):
#   add --no_use_eosin_encoder

# To adjust domain split (default 0.25 = 75% aligned):
#   add --case_b_prob 0.25
```

---

## GPU and Batch Size Recommendations

| GPU (VRAM) | Recommended batch_size | Notes |
|------------|----------------------|-------|
| T4 (16 GB) | 2–3 | Tight; may OOM with edge_encoder='v2' |
| A10G (24 GB) | 4–6 | Comfortable start; monitor with `nvidia-smi` |
| **A100 40GB** | **8** | **Recommended — matches script default** |
| A100 80GB | 12–16 | For faster convergence / larger batches |

**Start with batch_size=8 on A100 40GB.** Watch GPU VRAM in the first 5 steps with `nvidia-smi dmon -s u` in a second terminal. If VRAM > 90%, drop to 6.

Why UNI is the main memory consumer: for batch_size=8, UNI processes 8×16=128 sub-crops (128×[3,224,224]) through ViT-L/16 each step. It runs under `@torch.no_grad()` so no gradient memory, but forward activations are still ~3-4 GB.

**Hugging Face Token**: UNI weights are gated on MahmoodLab/uni. Set `HF_TOKEN` before the first run:
```bash
export HF_TOKEN=hf_xxxx
huggingface-cli login --token $HF_TOKEN
```

---

## Common Commands

```bash
# Train multi-stain (destaining)
python scripts/train/train_mist.py --data_dir /path/to/MIST --batch_size 8

# Resume from checkpoint
python scripts/train/train_mist.py --data_dir /path/to/MIST --resume_from checkpoints/destaining_v1/last.ckpt

# Evaluate
python scripts/eval/eval_mist.py --checkpoint checkpoints/destaining_v1/last.ckpt --data_dir /path/to/MIST
```

---

## Known Pitfalls

- **Folder name mismatch**: Zips extract as `HER2-Destained/` but code expects `HER2/`. Launch script renames automatically; if running manually, do `mv HER2-Destained HER2`.
- **HF access gate**: UNI model will fail to download without HF token + MahmoodLab/uni access approval.
- **Misalignment tolerance**: H&E and IHC slices are consecutive cuts, not the same section. The 75/25 routing and alignment-free losses in Case B handle this — do NOT add full-res pixel losses or spatially-sensitive losses (block DAB, FM loss) in the misaligned path.
- **Eosin dirs absent**: If `trainA-E/` does not exist, the dataset prints a warning and returns zero tensors for `he_e_map`. Training continues safely, but the EosinEncoder receives all-zero input. Either generate the E-maps or set `--no_use_eosin_encoder`.
- **Empty discriminator optimizer**: When all disc weights are 0 (sanity-check mode), a dummy `nn.Parameter` is registered so the Adam optimizer doesn't crash with an empty parameter list. This is intentional — don't remove it.
- **VRAM spikes**: The R1 gradient penalty (every 16 steps) temporarily allocates extra graph memory. If near VRAM limit, increase `r1_every` to 32 or set `r1_weight=0`.
- **proj_disc R1 cost**: The `ProjectionDiscriminator` also gets its own R1 penalty. If VRAM is tight, disable proj_disc entirely with `proj_disc_weight=0.0`.
- **num_workers on Lightning AI**: Default is 4. If you see `BrokenPipeError` in data loaders, set `num_workers=0` to debug.
- **wandb offline mode**: If network is restricted, add `WANDB_MODE=offline` before the training command.
- **DAB block loss at non-512 resolution**: `dab_block_size=32` assumes 512×512 input (gives 16×16 blocks). If you change `image_size`, adjust `dab_block_size` proportionally or set it to 0.

---

## Project Conventions
- Tensors in [-1, 1] throughout training (normalized with mean=0.5, std=0.5).
- H-maps are 1-channel grayscale, also in [-1, 1].
- UNI tokens are in original float32 scale (not normalized to [-1, 1]).
- Keep tensor/value-range handling consistent with existing code paths.
- Follow existing script style: argparse-driven CLIs with explicit defaults.
- Prefer extending current modules over adding duplicate implementations.

## Validation Expectations
- There is no dedicated automated test suite in this repo.
- Validate by running the smallest relevant train/eval path you changed.
- For metric-related changes, run corresponding eval script and verify output artifacts/metrics are produced.

## Change Boundaries
- If requested task is documentation/instructions-only, do not modify training/model code.
- For feature work, update only touched areas and keep script interfaces backward compatible unless asked otherwise.
