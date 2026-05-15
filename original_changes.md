# Original UNIStainNet vs Current Repo: Architecture and Loss Changes

This document compares the original paper implementation files you provided with the current repository branch in this workspace. It focuses on:

- what changed in the architecture,
- how label conditioning was originally used,
- how the current repo differs,
- what the added losses are doing,
- which of those added losses look useful and which look like candidates for removal,
- and how to reintroduce stain-label conditioning in a way that works beside the existing attention layer rather than replacing it.

The main conclusion is simple:

- the original implementation uses stain conditioning through SPADE + FiLM in the decoder,
- the current repo replaces that with Cross-Attention over UNI tokens and leaves the stain embedding unused,
- the best path is to keep Cross-Attention and add back explicit FiLM conditioning at every decoder stage.

---

## 1. High-Level Architectural Comparison

### Original implementation

The original paper implementation is a SPADE-UNet generator conditioned on two signals:

- UNI spatial features, injected through SPADE modulation,
- stain/class embedding, injected through FiLM modulation.

The core decoder block is the `SPADEBlock` in [original src/models/blocks.py](Papers/UNIStainNet%20Original%20Implementation/UNIStainNet-main/src/models/blocks.py).

That block does three things:

- normalizes decoder activations with InstanceNorm,
- computes spatial gamma/beta from UNI features via convolutional layers,
- computes channel-wise gamma/beta from the stain embedding via linear layers,
- combines them as:

$$
(\gamma_{spade} + \gamma_{film}) \cdot \mathrm{Norm}(x) + (\beta_{spade} + \beta_{film})
$$

So in the original design, the stain label is a first-class conditioning signal, not just metadata.

The bottleneck also contains `SelfAttention(512)` in the original code. It is a global context mixer at 16×16 and is part of the original generator definition.

### Current repo

The current repo moved the main semantic injection mechanism from SPADE+FiLM to token-level Cross-Attention. The current generator uses:

- a 1-channel H-map encoder,
- a bottleneck with `ResBlock + SelfAttention + ResBlock`,
- Cross-Attention blocks at multiple decoder scales,
- optional Eosin injection,
- optional edge encoder.

The important difference is that the current generator still creates `class_embed(labels)` but does not use it to control the output. That means the stain label path is effectively dead.

So the current repo has a stronger semantic mechanism in Cross-Attention, but it lost the original explicit stain-style control path.

---

## 2. Original vs Current: File-by-File Comparison

### `src/models/blocks.py`

#### Original

Contains:

- `SPADEBlock`
- `ResBlock`
- `SelfAttention`

`SPADEBlock` is the important original conditioning unit. It combines UNI spatial modulation and class embedding modulation.

`SelfAttention` is a standard bottleneck attention module over the same feature map.

#### Current repo

Contains:

- `SPADEBlock`
- `ResBlock`
- `SelfAttention`
- `CrossAttention`

The current repo adds `CrossAttention`, where Q comes from spatial decoder features and K/V come from UNI tokens. This is a semantic injection upgrade, but it does not replace the original need for label conditioning.

### `src/models/generator.py`

#### Original

The generator is a SPADE-UNet:

- encoder processes RGB H&E,
- bottleneck has `ResBlock -> SelfAttention -> ResBlock`,
- decoder uses `SPADEBlock` at each stage,
- `class_embed(labels)` is part of the forward pass,
- UNI features are converted to multi-scale spatial maps through `UNIFeatureProcessor` or `UNIFeatureProcessorHighRes`.

The original forward path is basically:

- encode H&E,
- process at bottleneck,
- decode with SPADE conditioned on UNI maps and stain embedding,
- output stained RGB image.

#### Current repo

The generator now:

- encodes a 1-channel H-map instead of RGB,
- uses Cross-Attention at decoder stages,
- still includes the bottleneck self-attention,
- optionally injects Eosin features,
- optionally injects edge features,
- but does not actually use the stain embedding after creating it.

This is the single biggest behavior change relevant to your staining-collapse problem.

### `src/models/trainer.py`

#### Original

The original training config uses a paper-style loss mix that is relatively simple:

- LPIPS at 128 and 256,
- H&E edge loss,
- low-res L1 for misaligned supervision,
- unconditional adversarial loss,
- DAB intensity,
- feature matching.

There is no big stack of DAB-specific auxiliary losses.

#### Current repo

The current trainer adds many extra losses and more discriminator variants:

- full-res aligned L1/LPIPS,
- spectral misalignment loss,
- DAB histogram matching,
- DAB block matching,
- DAB sparsity hinge,
- DAB contrast,
- IHC edge loss,
- H&E edge loss,
- background white loss,
- PatchNCE,
- crop discriminator,
- projection discriminator,
- unconditional discriminator,
- multi-scale discriminator.

This is a much more heavily regularized system than the original.

### `src/models/uni_processor.py`

#### Original

UNI tokens are converted to spatial feature maps and used for SPADE conditioning.

That matters because the original model expects spatial semantic maps, not just tokens.

#### Current repo

The processor still exists, but the main path now uses token Cross-Attention directly, so the original SPADE-style usage is no longer the central semantic path.

### `src/models/discriminator.py`

#### Original

Only PatchGAN + feature matching + hinge losses.

#### Current repo

Adds a projection discriminator and a multi-scale discriminator path, which makes the training more stain-aware in the adversarial branch, but does not fix the generator’s missing label conditioning.

### `src/utils/metrics.py`

#### Original

Already contains a fairly broad metric suite, including LPIPS, SSIM, PSNR, FID, KID, DAB metrics, IOD metrics, and downstream metrics.

#### Current repo

The validation path in trainer still mostly logs LPIPS, SSIM, and DAB MAE directly. The larger metric utility file is present but the live training loop only exposes a subset of it.

---

## 3. Self-Attention: What Changed and What It Means

The bottleneck self-attention exists in both implementations.

### Original purpose

The original `SelfAttention` is a 16×16 global mixer.

It helps the bottleneck summarize long-range spatial context before decoding.

### Current purpose

It is still present and still useful, but it is not the stain-conditioning mechanism.

### Important interpretation

- It does not separate HER2 vs Ki67 vs ER vs PR.
- It only mixes spatial context.
- The stain-specific signal has to come from FiLM / label conditioning, not from self-attention.

So self-attention is not the problem to remove. It is a supporting block, not the label controller.

---

## 4. What the Original Losses Were Doing

This section summarizes the losses used in the original train configuration and whether they are good ideas.

### 4.1 `lpips_weight = 1.0`

**Meaning:** perceptual similarity at the main low-resolution validation/training scale.

**Purpose:** preserve perceptual structure while tolerating weak misalignment.

**Assessment:** good idea. Keep it.

### 4.2 `lpips_256_weight = 0.5`

**Meaning:** a second LPIPS term at a finer resolution.

**Purpose:** adds extra perceptual pressure on mid-level details.

**Assessment:** good idea. Keep it if memory allows.

### 4.3 `he_edge_weight = 0.5`

**Meaning:** H&E edge structure preservation loss in the misaligned case.

**Purpose:** encourage the generator to retain structural cues from H&E.

**Assessment:** good idea if it is used only in the misaligned route. It is useful because it gives structure guidance that is not tied to exact pixel correspondence.

### 4.4 `l1_lowres_weight = 1.0`

**Meaning:** low-resolution pixel loss in the misaligned case.

**Purpose:** force coarse alignment and approximate stain/color behavior.

**Assessment:** borderline.

- Good idea if the data is only mildly misaligned.
- Better to replace with a translation-invariant frequency loss when the slice drift is large.

In the current repo, this has effectively been replaced by a spectral loss. That is a reasonable upgrade.

### 4.5 `adversarial_weight = 0.0`

**Meaning:** the main global adversarial branch is disabled in the paper config for MIST.

**Purpose:** avoid destabilizing training when the rest of the supervision is already strong.

**Assessment:** reasonable to keep low or off early in training. A pure GAN loss is not always necessary.

### 4.6 `uncond_disc_weight = 1.0`

**Meaning:** unconditional PatchGAN adversarial loss.

**Purpose:** make outputs look like real IHC globally, without depending on spatial pairing.

**Assessment:** good idea.

This is one of the most useful losses for style realism.

### 4.7 `dab_intensity_weight = 0.2`

**Meaning:** match top-10% DAB intensity.

**Purpose:** ensure the output has the correct amount of strong brown staining.

**Assessment:** good idea, but it is coarse on its own.

Use it as a global stain-strength anchor, not as the only DAB loss.

### 4.8 `dab_contrast_weight = 0.0`

**Meaning:** no explicit class-ordering DAB supervision in the original paper config.

**Assessment:** reasonable to keep off by default.

It can help if class-specific stain ordering is unstable, but it can also overconstrain the generator.

### 4.9 `dab_sharpness_weight = 0.0`

**Meaning:** no explicit DAB sharpness supervision.

**Assessment:** good to leave off unless you observe diffuse brown wash.

This can be useful, but it is not a first-priority loss.

### 4.10 `gram_style_weight = 0.0`

**Meaning:** no VGG Gram style loss.

**Assessment:** reasonable to keep off.

It is a texture prior, but it can push the generator toward generic texture matching rather than stain-specific realism.

### 4.11 `edge_weight = 0.0`

**Meaning:** no extra Fourier edge loss in the original config.

**Assessment:** optional.

If the network is already sharp enough, it is not necessary.

### 4.12 `crop_disc_weight = 0.0`

**Meaning:** no crop discriminator.

**Assessment:** optional.

Useful only if you need localized detail realism, but not required for the paper-style baseline.

### 4.13 `feat_match_weight = 10.0`

**Meaning:** feature matching from the unconditional discriminator.

**Purpose:** stabilize the generator and make its features resemble real IHC features.

**Assessment:** good idea.

This is a strong and sensible stabilizer.

### 4.14 `patchnce_weight = 0.0`

**Meaning:** PatchNCE is off in the original config.

**Assessment:** reasonable to keep off initially.

It can help with correspondence, but it is not required for a strong baseline and can become another balancing complication.

### 4.15 `bg_white_weight = 0.0`

**Meaning:** no background whitening loss.

**Assessment:** optional, usually not a priority.

Only add it if the model consistently stains background too much.

---

## 5. Current Repo Loss Additions: What They Mean and Whether to Keep Them

This section covers the new losses added after the original paper implementation.

### 5.1 `train/l1_fullres`

**Meaning:** aligned full-resolution L1 loss.

**Purpose:** very strong pixel supervision when the generator input and target are truly aligned.

**Assessment:** good idea only in the aligned case.

Do not use it in the misaligned branch.

### 5.2 `train/lpips_fullres`

**Meaning:** aligned full-resolution LPIPS.

**Purpose:** perceptual correction at full resolution.

**Assessment:** good idea in the aligned case.

### 5.3 `train/spectral_misalign`

**Meaning:** FFT magnitude loss for misaligned pairs.

**Purpose:** preserve frequency content without being punished by spatial shift.

**Assessment:** good idea.

This is a sensible replacement for low-res pixel losses when the sections are not exactly aligned.

### 5.4 `train/dab_histo`

**Meaning:** Wasserstein-1 between DAB intensity histograms.

**Purpose:** match the shape of the DAB distribution, not just the mean.

**Assessment:** good idea.

This is one of the better additions because it prevents trivial solutions that match only average stain.

### 5.5 `train/dab_block`

**Meaning:** per-block DAB mean matching.

**Purpose:** enforce local spatial placement of stain.

**Assessment:** good idea only in the aligned case.

Better to keep it limited to the aligned route.

### 5.6 `train/dab_sparsity`

**Meaning:** hinge penalty against over-staining.

**Purpose:** prevent the generator from painting too much DAB everywhere.

**Assessment:** useful, but should be treated carefully.

- Good if the model produces a brown wash.
- Risky if the true stain is genuinely dense in some classes.

So I would keep it as a conditional experiment, not a guaranteed default.

### 5.7 `train/dab_contrast`

**Meaning:** class-ordering loss for stain strength.

**Purpose:** try to enforce that stronger classes produce more DAB than weaker ones.

**Assessment:** mixed.

This can help if the stain ordering is unstable, but it can also overconstrain the generator and bake in an overly simplistic ordering.

I would keep it only if you have a real class-ordering problem.

### 5.8 `train/ihc_edge`

**Meaning:** edge loss between generated image and real IHC in the aligned case.

**Purpose:** sharpen membranes and boundaries.

**Assessment:** good idea in aligned training.

This is useful if the output is soft or blurry.

### 5.9 `train/he_edge`

**Meaning:** edge loss against H&E structure in the misaligned case.

**Purpose:** preserve structural continuity from input H&E.

**Assessment:** good idea in misaligned training.

It should stay out of the aligned branch.

### 5.10 `train/patchnce`

**Meaning:** contrastive feature consistency between H&E and generated output.

**Purpose:** preserve correspondence in feature space.

**Assessment:** optional.

Useful if you need better structure transfer, but not first priority.

### 5.11 `train/feat_match`

**Meaning:** discriminator feature matching.

**Purpose:** stabilize adversarial training.

**Assessment:** good idea in aligned supervision; mixed in misaligned supervision.

If used in the wrong branch, it can force the model to care too much about spatial correspondence that is not actually reliable.

### 5.12 `train/proj_adv_g` / `train/proj_adv_d`

**Meaning:** projection discriminator adversarial terms.

**Purpose:** make the discriminator stain-aware.

**Assessment:** good idea.

This is one of the best additions because it gives stain-specific realism pressure.

### 5.13 `train/uncond_adv_g` / `train/uncond_adv_d`

**Meaning:** unconditional adversarial loss terms.

**Purpose:** generic realism.

**Assessment:** good idea.

Keep this if the image quality needs adversarial sharpening.

### 5.14 `train/crop_adv_g` / `train/crop_adv_d`

**Meaning:** local crop discriminator losses.

**Purpose:** strengthen local texture realism.

**Assessment:** optional.

Useful for local detail, but not required unless patch-level texture is weak.

### 5.15 `train/r1_penalty`

**Meaning:** discriminator regularization.

**Purpose:** prevent the discriminator from becoming too sharp or unstable.

**Assessment:** good idea.

Keep it, especially if any GAN branch is enabled.

### 5.16 `train/bg_white`

**Meaning:** force background toward white.

**Purpose:** reduce unwanted background staining.

**Assessment:** optional.

Use only if the background is clearly problematic.

---

## 6. Recommended Architecture: Keep Attention, Add Back Label Conditioning

You asked specifically not to remove the attention layer. The best architecture is therefore a hybrid one.

### Recommended decoder block order

At each decoder stage, use:

1. upsample,
2. concatenate skip and edge signals,
3. conv,
4. Cross-Attention over UNI tokens,
5. SPADE + FiLM conditioning using UNI spatial maps and stain label,
6. activation.

That gives you both:

- semantic token-level conditioning from Cross-Attention,
- explicit stain control from FiLM,
- spatial tissue structure from SPADE.

### Why this is the right compromise

- Cross-Attention keeps the current repo’s semantic upgrade.
- FiLM restores the original stain-label mechanism.
- SPADE restores the original per-location semantic modulation.
- The self-attention bottleneck remains as a global context mixer.

This is the closest practical version of the original paper idea while preserving the later improvements already in the current branch.

### What must change in the forward pass

The current generator should do this:

- compute `class_emb = self.class_embed(labels)` and actually use it,
- compute `uni_spatial` maps from the UNI processor,
- keep `CrossAttention(x, uni_tokens)` in the decoder,
- pass `uni_spatial[scale]` and `class_emb` into a `SPADEBlock` or equivalent FiLM block at the same scale.

In other words, the label embedding should become a real control signal at every decoder level, not just an unused vector.

### What should not change

- do not remove Cross-Attention,
- do not remove bottleneck self-attention,
- do not remove the encoder/skip structure,
- do not remove the Eosin injection if it helps membrane topology.

---

## 7. Practical Priority Order for Future Changes

If I had to rank the next steps:

1. Reactivate stain-label conditioning in the decoder.
2. Keep Cross-Attention as semantic injection.
3. Keep the bottleneck self-attention.
4. Keep the good losses: LPIPS, feature matching, DAB intensity, DAB histogram, and the aligned/misaligned split.
5. Keep the new stain-aware discriminator, but do not let the model depend on too many weak auxiliary losses at once.

---

## 8. Bottom Line

The original implementation and the current repo are not minor variants. The current repo has moved from:

- SPADE + FiLM over UNI spatial maps and stain embedding,

to:

- Cross-Attention over UNI tokens with the stain embedding unused.

That explains why stain identity can collapse.

The best fix is not to remove attention. The best fix is to restore stain conditioning beside attention:

- use Cross-Attention for UNI semantics,
- use SPADE/FiLM for stain label control,
- keep the bottleneck self-attention,
- keep only the losses that directly help stain realism and structure without overconstraining the model.

If you want, I can next turn this into a concrete implementation plan with exact module-level edits, still without modifying any code.
