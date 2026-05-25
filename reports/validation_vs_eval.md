# Validation vs Evaluation: Detailed Behavior Report

Date: 2026-05-24
Scope: UNIStainNet training-time validation (Lightning) vs evaluation scripts (MIST/BCI)

This report explains, in code-level detail, what happens during training validation and evaluation runs, including inputs/outputs, how metrics are computed, and why results can differ even when both use random crops.

---

## 1) Training-time validation (inside the training loop)

### 1.1 When it runs
- Triggered at the end of each epoch by PyTorch Lightning.
- Uses the Lightning val dataloader defined in the DataModule.
- Runs on the validation split only.

### 1.2 Data path (MIST example)
Source: src/data/mist_dataset.py
- The Lightning DataModule is MISTMultiStainCropDataModule.
- In setup('fit'/'validate'/'test'): val_dataset = MISTMultiStainCropDataset(split='val').
- val_dataloader() returns a DataLoader over val_dataset.
- Each sample is built from the following directories:
  - valA (H&E RGB)
  - valA-H (H map)
  - valA-E (E map, optional; zeros if missing)
  - valB (IHC RGB)
  - valB-H (IHC H map)
- IMPORTANT: The dataset always does random cropping (512x512 by default), even when augment=False.

### 1.3 Input/Output per validation step
Source: src/models/trainer.py
- batch tuple:
  (he_rgb, ihc_rgb, he_h_map, ihc_h_map, he_e_map, labels, fnames)
- On-the-fly UNI extraction:
  - he_rgb is split into 4x4 sub-crops.
  - Each sub-crop is resized to 224x224 and normalized with ImageNet stats.
  - UNI features are extracted for each batch.

- Generator call (validation always uses EMA model):
  - generated = generator_ema(he_h_map, uni, labels, e_maps=he_e_map)
  - This is always Case B inference: input is H&E H map.

### 1.4 Metrics computed (training validation)
Source: src/models/trainer.py
- LPIPS (optional, default enabled):
  - Computed on downsampled images at size = image_size // 4.
  - For 512 images, LPIPS runs at 128x128.
  - lpips_val = lpips_fn(gen_lpips, ihc_lpips).mean().

- SSIM (optional, default enabled):
  - Computed on [0, 1] scaled tensors.
  - structural_similarity_index_measure(gen_01, real_01, data_range=1.0).

- DAB MAE (optional, default enabled):
  - Uses DABExtractor on CPU.
  - Computes top-10% mean intensity per image (p90 mean).
  - dab_mae = mean(|p90(gen) - p90(real)|).

- FID (optional, default disabled in current setup):
  - Torchmetrics FrechetInceptionDistance with feature=2048.
  - Streaming update per batch; computed at epoch end.

- KID (optional, default disabled):
  - Torchmetrics KernelInceptionDistance with feature=2048.
  - Streaming update per batch; computed at epoch end.

- UNI-FID (optional, default disabled):
  - Accumulates generated and real tensors for the entire epoch.
  - Computes Frechet distance in UNI feature space.
  - Uses MahmoodLab/uni model and CLS token features.

### 1.5 Output logging
- All metrics are logged to Lightning via self.log().
- Lightning aggregates per-batch logs across the entire validation epoch.
- If distributed, sync_dist=True ensures cross-GPU aggregation.

---

## 2) Evaluation scripts (standalone evaluation)

There are two primary evaluation entrypoints:
- scripts/eval/eval_mist.py
- scripts/eval/eval_bci.py

### 2.1 Which dataset split is used

MIST:
- eval_mist.py builds MISTMultiStainCropDataModule.
- Calls dm.setup('test'), then dm.test_dataloader().
- In MISTMultiStainCropDataModule, test_dataloader() == val_dataloader().
- Conclusion: For MIST, evaluation uses the val split, not a separate test split.

BCI:
- eval_bci.py builds BCICropDataModule and calls setup('test').
- BCICropDataModule uses HE/test and IHC/test for val_dataset.
- test_dataloader() returns val_dataloader().
- Conclusion: For BCI, evaluation uses the dataset’s test directories.

### 2.2 Input/Output per evaluation run (MIST)
Source: scripts/eval/eval_mist.py
- For each stain (HER2/Ki67/ER/PR):
  - Creates a per-stain DataModule with that stain only.
  - Iterates the test loader (which is val set).
  - Extracts UNI features on the fly from H&E.
  - Uses model.generate(...) to produce images.

- Each batch yields:
  - generated (gen), ihc_rgb (real), he_rgb (he), filenames.

### 2.3 Metrics computed in eval_mist
- FID and KID (always computed):
  - Torchmetrics FrechetInceptionDistance and KernelInceptionDistance.
  - Streaming updates per batch.

- LPIPS:
  - LearnedPerceptualImagePatchSimilarity on full resolution.

- SSIM / PSNR:
  - Torchmetrics StructuralSimilarityIndexMeasure and PeakSignalNoiseRatio.

- H&E structure metrics:
  - compute_he_structure_metrics (SSIM on Sobel edges).
  - compute_h_channel_ssim (SSIM on hematoxylin maps).
  - compute_nmi (normalized mutual information).

- DAB metrics:
  - p90 scores for gen and real.
  - per-pair histogram KL/JSD (ODA-GAN style).

- Optional UNI-FID:
  - compute_uni_fid if --skip_uni_fid is not passed.

### 2.4 Metrics computed in eval_bci
Source: scripts/eval/eval_bci.py
- Calls compute_image_quality_metrics (full-resolution, includes FID/KID/LPIPS/SSIM/PSNR).
- compute_dab_metrics (per-class breakdown).
- compute_iod_metrics (IOD/mIOD).
- Optional downstream classifier metrics (AUROC, SFS).
- Optional UNI-FID.

### 2.5 Output logging
- Evaluation writes a JSON report to eval_output/.
- It prints a summary to stdout.
- There is no Lightning aggregation; eval runs a standalone full sweep.

---

## 3) Why metrics can differ even when both use random crops

Random crops do contribute variance, but they are not the only reason for mismatched numbers. The main differences are:

1) Different metric definitions
- Training LPIPS is computed on downsampled images (image_size // 4).
- Eval LPIPS is computed at full resolution.
- DAB MAE in training uses only p90 mean; eval also uses histogram KL/JSD and pooled stats.

2) Different aggregation scheme
- Training logs per-batch metrics and Lightning averages them.
- Eval aggregates metrics explicitly across the dataset (FID/KID streaming, dataset-level averaging).

3) Different model usage and optional flags
- Training validation always uses the EMA generator.
- Eval uses model.generate() which may include classifier-free guidance (cfg) based on args.

4) Random crops are different on each run
- Both training and eval use random crops, so repeated runs can produce different results.
- But if you observe consistent, large differences between train-val and eval, the metric definitions and aggregation method are typically the dominant causes.

---

## 4) Input/Output summary (per step or per run)

Training validation (per batch):
- Input: he_rgb, ihc_rgb, he_h_map, ihc_h_map, he_e_map, labels
- Output: generated (EMA), per-batch metrics
- Aggregation: Lightning averages per-batch logs over all val batches

Evaluation (per run):
- Input: full val/test dataloader
- Output: generated images and dataset-level metrics
- Aggregation: explicit dataset-level metric accumulation

---

## 5) Comparison table

| Aspect | Training Validation | Evaluation (MIST) | Evaluation (BCI) |
|---|---|---|---|
| Dataset split | val | val (test_dataloader == val) | test folders (HE/test, IHC/test) |
| Cropping | random 512x512 | random 512x512 | random 512x512 |
| UNI features | on-the-fly | on-the-fly | on-the-fly |
| Generator used | EMA | EMA (via model.generate) | EMA (via model.generate) |
| LPIPS | downsampled (image_size // 4) | full-res | full-res |
| SSIM | per-batch | per-batch, aggregated | per-batch, aggregated |
| DAB | p90 MAE only | p90 + KL/JSD stats | full DAB metrics + per-class |
| FID/KID | optional, default off | on by default | on by default |
| UNI-FID | optional, default off | optional | optional |
| Aggregation | Lightning averages per-batch logs | explicit dataset-level accumulation | explicit dataset-level accumulation |
| Output | logged to logger | JSON + printed summary | JSON + printed summary |

---

## 6) Practical takeaways

- If you want train-val and eval numbers to match, you need to match:
  - crop determinism (e.g., center crop),
  - metric definitions (same resolution, same DAB computation),
  - aggregation scheme (dataset-level vs per-batch average).
- Otherwise, differences are expected and can be large.
