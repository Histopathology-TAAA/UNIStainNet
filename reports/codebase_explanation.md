# UNIStainNet — Codebase Explanation

This document explains the repository components from the **generator** onward, focusing on purpose, data I/O shapes, architecture correspondence, and where to edit common behaviours. Use this as a quick reference when editing or extending the project.

---

## Quick global notes
- Primary frameworks: PyTorch + PyTorch Lightning.
- Tensor ranges: most model I/O uses `[-1, 1]` (RGB and H/E maps). UNI ViT inputs use ImageNet normalization and 224×224 crops.
- Central MIST batch contract (7-tuple): `(he_rgb, ihc_rgb, he_h_map, ihc_h_map, he_e_map, stain_label, filename)` — shapes usually `[B, C, H, W]` with `H=W=512` (or `1024` for 1024 pipelines).

---

## Data contract (canonical)
- he_rgb: [B, 3, H, W], range [-1,1]
- ihc_rgb: [B, 3, H, W], range [-1,1]
- he_h_map: [B, 1, H, W], range [-1,1]
- ihc_h_map: [B, 1, H, W], range [-1,1]
- he_e_map: [B, 1, H, W], range [-1,1] (zeros if absent)
- stain_label: [B] long tensor (label index)
- filename: list[str]

Always update all callers (trainer, eval, debug, hf_space) if this contract changes.

---

## `src/models/generator.py` — SPADEUNetGenerator
- Purpose: maps a Hematoxylin map (H-map) + UNI spatial tokens + optional E-map + stain embedding → generated IHC RGB.
- Key concepts:
  - H-map encoder (1-channel input) builds spatial features at multiple scales.
  - Decoder uses SPADE/FILM style normalization conditioned by UNI tokens (injected via Cross-Attention at decoder levels).
  - Optional Eosin encoder injects eosin features at bottleneck.
- Important methods:
  - `forward(h_maps, uni_tokens, labels, e_maps=None)` → `[B, 3, H, W]` ([-1,1]).
  - `encode()` returns encoder feature maps used by PatchNCE or intermediate losses.
  - `generate()` wrapper used in eval/demo that handles CFG (guidance_scale), seed, ema selection and background composite.
- Inputs/outputs: follow the canonical batch contract but `forward` expects only `he_h_map` (1ch) + `uni` tokens; generation functions wrap the rest.
- Edit guidance:
  - To change CFG behavior or guidance defaults, edit `generate()` in this file or trainer wrapper.
  - To change how UNI is injected (e.g., alter cross-attention heads), edit connections in decoder blocks here and inspect `src/models/blocks.py`.

---

## `src/models/blocks.py` — primitive blocks
- Contains: `ResBlock`, `SPADEBlock`, `CrossAttention`, `SelfAttention`, `SPADE`-related layers and helpers.
- Purpose: building blocks for generator and attention mechanisms.
- Notes:
  - CrossAttention expects channels divisible by `n_heads`.
  - SPADE blocks apply spatially-adaptive normalization conditioned on tokens or maps.
- Edit guidance:
  - Keep numerical compatibility (channel divisibility) when changing number of heads or feature sizes.

---

## `src/models/uni_processor.py` — UNI token processors
- Purpose: convert UNI ViT patch tokens into multi-scale spatial tensors consumed by SPADE (e.g., map 1024-d tokens → spatial features of shape `[B, S*S, 1024]` then transform to `B, C, H, W`).
- Implementations: `UNIFeatureProcessor`, `UNIFeatureProcessorHighRes` (different upsampling / projection strategies).
- Edit guidance:
  - Changes here must match the `uni_spatial_size` expected by `UNIStainNetTrainer` and generator injection points.

---

## `src/models/trainer.py` — `UNIStainNetTrainer` (Lightning Module)
- Purpose: orchestrates training, losses, optimizers, EMA, validation logging and `generate()` inference.
- Key behaviours:
  - Manual optimization: `automatic_optimization=False` (trainer performs its own gen/disc steps). Do not set `Trainer(accumulate_grad_batches=...)` — accumulation handled inside the module.
  - On-the-fly UNI extraction: optional `extract_uni_on_the_fly` flag; trainer can call `timm` UNI to generate tokens per-batch.
  - Loss aggregation: combines LPIPS, L1 (lowres/fullres), DAB-related losses, spectral loss, feature matching, adversarial losses, patchNCE, etc.
  - EMA: generator EMA stored & used for validation/inference.
- Important interfaces:
  - `training_step()` performs manual gen/disc alternation, respects `adversarial_start_step` and `r1_every`.
  - `validation_step()` uses `generate()` to produce validation images and metrics.
  - `generate(h_map, uni, labels, e_maps=None, guidance_scale=1.0, seed=None)` is the public inference API used by eval scripts and demo.
- Edit guidance:
  - To change optimizer/lr defaults, update trainer hparams in the training script or pass new defaults when instantiating.
  - To adjust accumulation behavior change the trainer parameter (e.g., `accum_steps`) inside this class; keep Lightning `accumulate_grad_batches` unset.
  - Changing the loss suite requires touching both `trainer.py` and `src/models/losses.py`.

---

## `src/models/discriminator.py`
- Contents: PatchGAN-style discriminators, ProjectionDiscriminator (stain-conditioned), MultiScale wrappers, R1 loss helpers.
- Edit guidance:
  - Discriminators are instantiated conditionally based on loss weights passed into `UNIStainNetTrainer`. If disabling, set the corresponding weights to `0.0` in training script.

---

## `src/models/edge_encoder.py`
- Purpose: edge/structure pathway (edge encoder v1/v2) and optional eosin encoder. Produces structural features injected at bottleneck.
- If you change its output shape, update generator and trainer expectation.

---

## `src/models/losses.py`
- Contains: VGG feature extractor for perceptual loss, Gram (style) calculations, PatchNCE contrastive loss, spectral loss utilities.
- Edit guidance: changing perceptual layers or weights must be mirrored where weights/hyperparams are set (train scripts or trainer hparams).

---

## `src/data/mist_dataset.py` and `src/data/bci_dataset.py`
- Purpose: dataset loaders, cropping/tiling pipelines, pairing rules, and UNI sub-crop preparation utilities.
- MIST specifics: multi-stain directory layout, pairing by filename stem, optional `trainA-E/` eosin channel directories.
- Output contract: the 7-element tuple described earlier.
- Edit guidance:
  - If adding new dataset fields (e.g., masks), update all callers (`trainer`, `eval`, `hf_space/app.py`, debug scripts) at once.

---

## `src/utils/dab.py` — DAB deconvolution
- Purpose: Ruifrok & Johnston stain matrix for DAB/Hematoxylin extraction and helpers for p90/intensity/histogram computation.
- Function: `extract_dab_intensity(images, normalize)` returns `[B, H, W]` DAB intensity maps.
- Edit guidance: If changing stain matrix or normalization, update metrics and losses that assume the same scale.

---

## `src/utils/metrics.py`
- Purpose: all evaluation metrics and helpers.
- Important metrics implemented:
  - Image quality: FID (Inception), KID, LPIPS, SSIM, PSNR (`compute_image_quality_metrics`).
  - Structural metrics: `compute_he_structure_metrics` (edge-SSIM/structure), `compute_h_channel_ssim` (H-channel SSIM), `compute_nmi` (normalized mutual information between H-map and generated image).
  - DAB metrics: `compute_dab_metrics` (p90, KL, JSD, Pearson), `compute_iod_metrics`.
  - UNI-FID: `compute_uni_fid` (optional expensive metric).
  - Helpers: `save_sample_grid`, `composite_background`.
- Edit guidance: Add/remove metrics here; eval scripts import specific functions. Keep heavy metrics (UNI-FID) optional behind flags.

---

## `scripts/train/*` (entrypoints)
- `scripts/train/train_mist.py` — 512px MIST training (random crops). Centralizes `LOSS_WEIGHTS` and instantiates `UNIStainNetTrainer` with many hparams. Uses Lightning `Trainer` with GPU.
- `scripts/train/train_mist_1024.py` — native 1024 training (no cropping). Larger memory usage.
- `scripts/train/train_bci.py` — BCI training config (paper settings).
- Edit guidance:
  - To change defaults for experiments, update the `LOSS_WEIGHTS` map or hparams passed to `UNIStainNetTrainer` in these scripts.
  - Remember to change `uni_spatial_size` consistently in training and eval/demo.

---

## `scripts/eval/*` (evaluation)
- `scripts/eval/eval_mist.py` — per-stain evaluator using streaming generation (memory-safe). Extracts UNI on-the-fly, yields per-batch to compute metrics incrementally. Recommended for limited RAM.
- `scripts/eval/eval_mist_1024.py` — 1024 evaluator; currently collects all outputs in memory (heavy), computes same metrics as above.
- `scripts/eval/eval_bci.py` — BCI evaluator; collects outputs and computes per-class and overall metrics, downstream probes, and UNI-FID optional.
- Edit guidance:
  - Convert `eval_mist_1024.py` and `eval_bci.py` to streaming by adopting the generator pattern used in `eval_mist.py` (replace `all_gen.append(...)` with per-batch yielding and incremental metric updates).
  - Use `--skip_uni_fid` and `--skip_downstream` flags to avoid expensive computations when desired.

---

## `scripts/debug/*`
- `scripts/debug/validate_data.py` — iterate a few real batches (CPU) and print shapes/ranges. Useful for folder layout and basic loader sanity.
- `scripts/debug/sanity_check_mist_dummy.py` — runs a single training + validation step on synthetic data and avoids loading UNI by patching trainer extraction to a fake random extractor. Good quick model-shape check.

---

## `hf_space/app.py` — Gradio demo
- Purpose: public demo for interactive generation and gallery browsing.
- Key behaviours:
  - Lazy model loading: downloads checkpoint or uses local `checkpoints/mist_multistain_last.ckpt` and loads `MahmoodLab/uni` on demand.
  - GPU-aware: tries to use HF Spaces ZeroGPU; falls back to gallery-only if no GPU.
  - Uses same preprocessing and UNI extraction pipeline as eval scripts.
- Edit guidance:
  - To change checkpoint source, set `MODEL_REPO` env var or modify `MODEL_REPO` constant.
  - To change gallery content, edit `hf_space/gallery/metadata.json` and images under `hf_space/gallery/images`.

---

## Edit checklist (practical quick edits)
- Change loss weights: update `LOSS_WEIGHTS` or pass new hparams in `scripts/train/train_mist.py` (and 1024/bci variants).
- Toggle eosin encoder: train flags `--use_eosin_encoder` / `--eosin_multi_scale` (and `edge_encoder`), implemented in `src/models/edge_encoder.py` + `generator.py`.
- Change UNI spatial size: update `uni_spatial_size` in trainer init; keep eval/demo consistent.
- Disable GAN: set `uncond_disc_weight=0.0`, `proj_disc_weight=0.0`, and `adversarial_weight=0.0` in training script.
- Convert evals to streaming: adapt the `generate_for_stain()` generator pattern from `scripts/eval/eval_mist.py`.
- Change `generate()` behaviour: edit `src/models/trainer.py`.
- Modify DAB extraction: edit `src/utils/dab.py` (careful: affects losses and metrics).

---

## Quick commands
- Validate data (CPU):
```bash
python scripts/debug/validate_data.py --data_dir /path/to/MIST --stains HER2 ER Ki67 PR --batch_size 4 --n_batches 3
```
- Sanity model loop (CPU):
```bash
python scripts/debug/sanity_check_mist_dummy.py
```
- Streaming eval (memory-safe):
```bash
python scripts/eval/eval_mist.py --checkpoint checkpoints/mist_multistain/last.ckpt --data_dir /path/to/MIST --batch_size 8 --skip_uni_fid
```

---

## Where to look when something breaks
- Shapes / data: `src/data/mist_dataset.py` and `scripts/debug/validate_data.py` (run first).
- Model forward / CFG: `src/models/generator.py` and `src/models/trainer.py` (`generate()` + `training_step`).
- Losses: `src/models/losses.py` and trainer weight wiring in `src/models/trainer.py`.
- Heavy metrics / optional components: `src/utils/metrics.py` (toggle with flags in eval scripts).

---

If you want, I can:
- Convert `scripts/eval/eval_mist_1024.py` and `scripts/eval/eval_bci.py` to streaming mode now (safe for low-RAM machines). 
- Or make a targeted change from the edit checklist — tell me which one and I will apply the patch and run a quick compile/test.


---

Generated by the developer assistant as `reports/codebase_explanation.md`.
