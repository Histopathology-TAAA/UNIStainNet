# Losses Through History

This report tracks how the UNIStainNet loss setup evolved from the original implementation to the current destaining branch.

## How To Read The Losses

The training loop uses two supervision modes:

| Case | Meaning | When it is used | How the loss is interpreted |
|---|---|---|---|
| Case A | Aligned supervision | Generator input is the IHC H-map | Pixel-level and spatial losses are valid because input and target are aligned enough |
| Case B | Misaligned supervision | Generator input is the H&E H-map | Pixel-level supervision is unsafe, so the code prefers translation-robust or lower-resolution losses |
| Both | Alignment-free supervision | Always active if the weight is non-zero | Distribution, feature, and adversarial losses that do not require exact pixel correspondence |

## Loss Glossary

| Loss | What it measures | Case | Calculation idea |
|---|---|---|---|
| `lpips_weight` | Perceptual similarity at the main scale | Both, but especially important in Case B | LPIPS between generated and target images after downsampling to the main evaluation size |
| `lpips_256_weight` | Finer perceptual similarity | Both | LPIPS on a 256 px version for 512 px training, or 512 px for 1024 px training |
| `lpips_512_weight` | Full-resolution perceptual similarity | Case A only in practice | LPIPS at full resolution, expensive and more sensitive |
| `l1_fullres_weight` | Pixel reconstruction error | Case A | Direct L1 between generated and target RGB images |
| `l1_lowres_weight` | Lower-resolution pixel error | Case B | L1 on a downsampled representation when exact alignment is not trusted |
| `he_edge_weight` | H&E structure preservation | Case B preferred | Sobel edge matching against H&E-derived structure |
| `ihc_edge_weight` | IHC structure preservation | Case A | Sobel edge matching against real IHC |
| `adversarial_weight` | Standard GAN realism pressure | Both | Hinge GAN loss from the global discriminator |
| `uncond_disc_weight` | Unconditional realism | Both | Discriminator sees only the generated IHC, not the stain label or H&E input |
| `feat_match_weight` | Texture/statistics matching | Case A | L1 distance between discriminator feature maps for real and fake |
| `dab_intensity_weight` | Overall DAB strength | Both | Matches the mean of the top DAB pixels, usually the top 10 percent |
| `dab_contrast_weight` | Stain-grade ordering | Both | Hinge-style ordering loss so 3+ > 2+ > 1+ > 0 |
| `dab_histo_weight` | DAB distribution shape | Both | Wasserstein-style distance between sorted DAB optical-density values |
| `dab_block_weight` | Local DAB placement | Case A | Blockwise average DAB matching with average pooling over fixed blocks |
| `dab_sparsity_weight` | Over-staining penalty | Both | Hinge loss on generated DAB mass being larger than real mass by a margin |
| `proj_disc_weight` | Stain-conditioned realism | Both | Projection discriminator conditioned on stain label |
| `gram_style_weight` | Texture style statistics | Both | Gram-matrix loss on VGG features |
| `patchnce_weight` | Cross-image feature consistency | Both | Patch-wise contrastive loss between input and output encoder features |
| `bg_white_weight` | Background whitening | Both | Encourages background regions to stay close to white |
| `edge_weight` | Generic edge sharpness | Both | Additional edge-based structural loss if enabled |

## Version Timeline

### 1. Original baseline implementation

The original code in `UNIStainNet-main` already had a solid but simpler loss set.

Active weights in that snapshot:

| Loss | Weight |
|---|---:|
| `lpips_weight` | 1.0 |
| `lpips_256_weight` | 0.5 |
| `lpips_512_weight` | 0.0 |
| `he_edge_weight` | 0.5 |
| `l1_lowres_weight` | 1.0 |
| `adversarial_weight` | 0.0 |
| `uncond_disc_weight` | 1.0 |
| `dab_intensity_weight` | 0.2 |
| `dab_contrast_weight` | 0.0 |
| `dab_sharpness_weight` | 0.0 |
| `gram_style_weight` | 0.0 |
| `edge_weight` | 0.0 |
| `crop_disc_weight` | 0.0 |
| `feat_match_weight` | 10.0 |
| `patchnce_weight` | 0.0 |
| `bg_white_weight` | 0.0 |

What this meant:
- The model relied on perceptual loss, a low-resolution pixel loss, H&E edge preservation, unconditional adversarial realism, DAB intensity, and feature matching.
- There was no explicit case-aware A/B split in the script.
- The loss design was still mostly coarse and did not yet include histogram, block-level DAB, sparsity, or stain-conditioned discrimination.

### 2. V0 - first destaining refactor

This was the first branch-level move toward the current structure-semantic setup.

Main change:
- The training logic became explicitly mixed-domain.
- Case A used aligned supervision.
- Case B used misaligned supervision.

Loss behavior:
- Case A used full-resolution pixel/perceptual supervision.
- Case B used lower-resolution and translation-robust supervision.
- The original DAB intensity and feature matching ideas were still kept.

### 3. V1 - loss logic corrected by case

This is where the branch became more careful about which loss is valid in which case.

Main changes:
- Case B stopped using pixel-sensitive supervision.
- Frequency-domain supervision replaced the old low-resolution pixel penalty for misaligned data.
- Feature matching was restricted to aligned supervision.
- The aligned/misaligned split was made configurable and stabilized around a 75/25 style mix.

Why this mattered:
- The model should not be punished for spatial drift that comes from consecutive tissue cuts.
- Case B had to stay alignment-robust.

### 4. V2 - Eosin bottleneck injection

This version changed the architecture, not the loss philosophy.

Main change:
- Eosin was added at the bottleneck so the generator could see membrane-related context.

Loss impact:
- No major new loss family was introduced here.
- The existing supervision remained the same, but the model had more structural input.

### 5. V3 - DAB distribution and stain-conditioned realism

This is the first big jump in loss design.

New losses added:
- `ihc_edge_weight = 0.1`
- `dab_histo_weight = 0.3`
- `dab_block_weight = 0.2`
- `proj_disc_weight = 1.0`

What they did:
- `ihc_edge_weight` added aligned IHC edge supervision in Case A.
- `dab_histo_weight` matched the full DAB distribution, not only its mean.
- `dab_block_weight` forced the stain to appear in the right places spatially.
- `proj_disc_weight` made the discriminator aware of stain identity so HER2 and nuclear stains could be judged differently.

This version is the point where the training objective became much more stain-aware.

### 6. V4 - tightened weights and sparsity control

This is the current version in the branch you are using.

Main changes over V3:
- `dab_block_weight` increased from 0.2 to 0.5.
- `proj_disc_weight` increased from 1.0 to 2.0.
- `dab_sparsity_weight = 0.3` was added.
- `dab_sparsity_margin = 0.05` was added.

Why:
- The model was still over-expressing stain in some stains.
- Block loss plateaued, so stronger local spatial supervision was needed.
- The projection discriminator needed more pressure to reduce the HER2 nuclear shortcut.
- The sparsity hinge added a direct cap on total DAB mass.

## Weight Comparison By Version

| Loss | Original baseline | V0 | V1 | V2 | V3 | V4 / current |
|---|---:|---:|---:|---:|---:|---:|
| `lpips_weight` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `lpips_256_weight` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| `lpips_512_weight` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `he_edge_weight` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| `l1_lowres_weight` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `adversarial_weight` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `uncond_disc_weight` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `dab_intensity_weight` | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| `dab_contrast_weight` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `dab_sharpness_weight` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `gram_style_weight` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `edge_weight` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `crop_disc_weight` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `feat_match_weight` | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 |
| `patchnce_weight` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `bg_white_weight` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `ihc_edge_weight` | - | - | - | - | 0.1 | 0.1 |
| `dab_histo_weight` | - | - | - | - | 0.3 | 0.3 |
| `dab_block_weight` | - | - | - | - | 0.2 | 0.5 |
| `proj_disc_weight` | - | - | - | - | 1.0 | 2.0 |
| `dab_sparsity_weight` | - | - | - | - | - | 0.3 |
| `dab_sparsity_margin` | - | - | - | - | - | 0.05 |

## Short Takeaway

- The earliest implementation was already using perceptual loss, weak pixel loss, DAB intensity, and feature matching.
- The first major shift was making the supervision case-aware.
- The second major shift was adding DAB distribution and location losses plus a stain-conditioned discriminator.
- The current version mainly tightens DAB control rather than changing the whole objective.

## Recommended Reading Order In The Code

1. [scripts/train/train_mist.py](../scripts/train/train_mist.py)
2. [src/models/trainer.py](../src/models/trainer.py)
3. [src/utils/metrics.py](../src/utils/metrics.py)
4. [src/utils/dab.py](../src/utils/dab.py)
5. [changes.md](../changes.md)

