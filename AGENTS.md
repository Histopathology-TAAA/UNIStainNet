# AGENTS.md

Agent instructions for this repository.

## Scope
- Applies to the entire repository.
- Keep changes focused and minimal. Do not refactor unrelated modules.
- Active branch: `exp-destaining-v1-attention` — destaining variant is the main work.

---

## First Read
- Project overview, setup, dataset layout, and baseline commands: [README.md](README.md)
- Architecture changelog (destaining refactor): [changes.md](changes.md)
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
| [src/models/generator.py](src/models/generator.py) | SPADEUNetGenerator — 1ch H-map encoder + CrossAttention decoder |
| [src/models/blocks.py](src/models/blocks.py) | CrossAttention, SelfAttention, ResBlock, SPADEBlock |
| [src/models/edge_encoder.py](src/models/edge_encoder.py) | EdgeEncoder (v1) / MultiScaleEdgeEncoder (v2) — parallel structure pathway |
| [src/models/uni_processor.py](src/models/uni_processor.py) | UNIFeatureProcessor / UNIFeatureProcessorHighRes — UNI token pipeline |
| [src/models/discriminator.py](src/models/discriminator.py) | PatchGAN + MultiScaleDiscriminator + loss functions |
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

### Mixed-Domain 50/50 Routing (trainer.py:625)
Each batch randomly picks one of two modes:

| Mode | Generator Input | Losses Applied |
|------|----------------|----------------|
| **Aligned (Case A)** | IHC H-map | Full-res L1 + full-res LPIPS vs IHC RGB |
| **Misaligned (Case B)** | H&E H-map | L1 @ 64×64 + LPIPS @ 128 & 256 (tolerates slice misalignment) |

UNI tokens are **always extracted from H&E RGB** regardless of mode.
Validation **always** uses H&E H-map → IHC RGB.

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

## Dataset Layout (Destaining)

### Required Folder Structure

```
<data_dir>/
  HER2/
    trainA/       ← H&E RGB images (paired)
    trainA-H/     ← H&E Hematoxylin channel (grayscale, same stems as trainA)
    trainB/       ← IHC RGB images (paired with trainA)
    trainB-H/     ← IHC Hematoxylin channel (grayscale, same stems as trainB)
    valA/
    valA-H/
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

### Extra Folders (Ignored by Current Code)
`trainA-E` (Eosin channel), `trainB-DAB`, `valA-E`, `valB-DAB` — present in the zips but not loaded. Available for future loss extensions.

---

## Lightning AI Setup

### File Layout Assumption
```
~/teamspace/studios/this_studio/
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
cd ~/teamspace/studios/this_studio/UNISTAINNET
export PYTHONPATH=$PWD
python scripts/train/train_mist.py \
    --data_dir  ~/teamspace/studios/this_studio/data/MIST \
    --stains    HER2 ER Ki67 PR \
    --batch_size 8 \
    --max_epochs 100 \
    --ckpt_dir  ~/teamspace/studios/this_studio/checkpoints/destaining_v1 \
    --wandb_name destaining_v1_attention
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
- **Misalignment tolerance**: H&E and IHC slices are consecutive cuts, not the same section. The 50/50 routing and downsampled losses in Case B handle this — do NOT add full-res pixel losses in the misaligned path.
- **VRAM spikes**: The R1 gradient penalty (every 16 steps) temporarily allocates extra graph memory. If you're near the VRAM limit, reduce `r1_every` to 32 or disable R1.
- **num_workers on Lightning AI**: Default is 4. If you see `BrokenPipeError` in data loaders, set `num_workers=0` to debug.
- **wandb offline mode**: If network is restricted, add `WANDB_MODE=offline` before the training command.

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
