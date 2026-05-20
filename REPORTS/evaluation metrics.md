# Evaluation Metrics

This report summarizes the metrics currently computed by the MIST evaluation scripts and clarifies what each metric compares.

## Pairing used in evaluation

For the standard image-quality, DAB, IOD, and UNI-FID metrics, the code compares:

- **Generated IHC vs real IHC**

For the structure metric added in this update, the code compares:

- **Generated IHC vs H&E reference**

The structure metric is intended to answer whether the generated stain preserves tissue layout and boundary structure seen in the input H&E image.

## Metrics already implemented

### Image quality metrics

- **`fid_inception`**
  - What it means: Fréchet Inception Distance.
  - How it is computed: Inception features are extracted from all generated and real IHC images, then the distance between the two feature distributions is calculated.
  - Interpretation: Lower is better.
  - Pair used: **generated IHC vs real IHC**.

- **`kid_mean_x1000`**
  - What it means: Kernel Inception Distance.
  - How it is computed: Unbiased MMD-based distance on Inception features, reported in the code as the mean multiplied by 1000.
  - Interpretation: Lower is better.
  - Pair used: **generated IHC vs real IHC**.

- **`lpips_mean`**
  - What it means: LPIPS perceptual distance at full resolution.
  - How it is computed: LPIPS network (`alex`) is applied directly to generated and real images in `[-1, 1]`.
  - Interpretation: Lower is better.
  - Pair used: **generated IHC vs real IHC**.

- **`lpips_128_mean`**
  - What it means: LPIPS computed after downsampling to 128×128.
  - How it is computed: Both images are bilinearly resized to 128×128 and then passed through LPIPS.
  - Interpretation: Lower is better.
  - Pair used: **generated IHC vs real IHC**.

- **`ssim_mean`**
  - What it means: Structural Similarity Index.
  - How it is computed: TorchMetrics SSIM on generated and real IHC in `[0, 1]`.
  - Interpretation: Higher is better.
  - Pair used: **generated IHC vs real IHC**.

- **`psnr_mean`**
  - What it means: Peak Signal-to-Noise Ratio.
  - How it is computed: TorchMetrics PSNR on generated and real IHC in `[0, 1]`.
  - Interpretation: Higher is better.
  - Pair used: **generated IHC vs real IHC**.

### DAB metrics

- **`dab_mae_overall`**
  - What it means: Absolute error between the canonical DAB p90 scores.
  - How it is computed: DAB intensity is deconvolved, the mean of the top 10% pixels is taken per image, then the mean absolute difference is computed.
  - Interpretation: Lower is better.
  - Pair used: **generated IHC vs real IHC**.

- **`dab_pearson_r`**
  - What it means: Correlation between generated and real DAB p90 scores.
  - How it is computed: Pearson correlation over the per-image p90 scores.
  - Interpretation: Higher is better.
  - Pair used: **generated IHC vs real IHC**.

- **`dab_kl`**
  - What it means: KL divergence between DAB histograms.
  - How it is computed: Each generated/real pair is converted to a 256-bin DAB histogram and KL is averaged across pairs.
  - Interpretation: Lower is better.
  - Pair used: **generated IHC vs real IHC**.

- **`dab_jsd`**
  - What it means: Jensen-Shannon divergence between DAB histograms.
  - How it is computed: Same 256-bin pairwise DAB histograms as above, with JSD averaged across pairs.
  - Interpretation: Lower is better.
  - Pair used: **generated IHC vs real IHC**.

### Optical density metrics

- **`miod_diff`**, **`miod_abs_diff`**, **`iod_diff`**, **`mfod_diff`**, etc.
  - What they mean: Differences in optical density summaries.
  - How they are computed: Images are converted to optical density using Beer-Lambert style conversion, then mean OD / integrated OD / fractional OD are computed.
  - Interpretation: Lower absolute differences are better.
  - Pair used: **generated IHC vs real IHC**.

### UNI-FID

- **`fid_uni`**
  - What it means: Fréchet distance in UNI feature space.
  - How it is computed: UNI CLS-token features are extracted for generated and real IHC images, then their distribution distance is measured.
  - Interpretation: Lower is better.
  - Pair used: **generated IHC vs real IHC**.

## Newly added structure metric

### `he_structure_ssim`

- What it means: Edge-based structural similarity between the generated IHC and the H&E input.
- How it is computed:
  1. Both images are converted to grayscale.
  2. Images are resized to 256×256.
  3. Sobel gradients are computed to obtain an edge magnitude map for each image.
  4. Each edge map is normalized to `[0, 1]` per image.
  5. SSIM is computed between the normalized edge maps.
- Interpretation: Higher is better.
- Pair used: **generated IHC vs H&E**.
- Why it is useful: It measures whether the generated stain preserves tissue structure from the H&E image while ignoring stain color differences.

## Where it is logged

The metric is computed and stored in the MIST eval outputs as:

- `stain_results['structure']['he_structure_ssim']`
- macro-averaged under the same key in the final summary

It is logged by:

- [scripts/eval/eval_mist.py](../scripts/eval/eval_mist.py)
- [scripts/eval/eval_mist_1024.py](../scripts/eval/eval_mist_1024.py)

The shared implementation lives in:

- [src/utils/metrics.py](../src/utils/metrics.py)

## Practical reading guide

- Use **FID / KID / LPIPS / SSIM / PSNR** to compare generated IHC to real IHC.
- Use **DAB metrics** to judge stain intensity and distribution realism.
- Use **IOD metrics** to compare optical density behavior.
- Use **`he_structure_ssim`** to judge whether the generated IHC still follows the tissue structure seen in H&E.

## Recommended interpretation for experiments

For ER / PR experiments, the most useful combination is usually:

- `fid_inception`
- `lpips_128_mean`
- `dab_mae_overall`
- `dab_pearson_r`
- `he_structure_ssim`

This gives a mix of:

- appearance similarity to real IHC,
- stain-intensity correctness,
- and structure preservation relative to H&E.
