# Attention and SPADE Flow in the Current UNIStainNet

This report explains the attention mechanism used in the current generator, how it connects to the SPADE block, and what every layer does.

The goal is to make the decoder behave like this:

- the convolution path builds a strong spatial feature map,
- the cross-attention path injects UNI semantic context,
- the SPADE block modulates the feature map using either UNI spatial maps or the attention output,
- the residual path can add the attention correction independently of SPADE.

## 1. What the generator is doing overall

The current generator is a structure-semantic decoupled UNet.

- The input `h_maps` is a 1-channel H-map with shape `[B, 1, 512, 512]` in the current default setting.
- The encoder compresses this structural input to a bottleneck feature map.
- The decoder upsamples step by step.
- At every decoder stage, the model can use three different semantic sources:
  - `uni_features` through cross-attention,
  - `uni_maps[...]` through SPADE,
  - `class_emb` through FiLM inside SPADE.

The important part is that the attention branch and the SPADE branch are related, but they are not the same branch.

## 2. Current default tensor shapes

### 2.1 Input and semantic tensors

| Tensor | Shape | Purpose |
|---|---|---|
| `h_maps` | `[B, 1, 512, 512]` | Structural input: Hematoxylin map only |
| `uni_features` | `[B, N, 1024]` | Raw UNI tokens used by cross-attention |
| `labels` | `[B]` | Stain labels used for FiLM conditioning |
| `class_emb` | `[B, 64]` | Learned embedding of the stain label |
| `uni_maps[32]` | `[B, 512, 32, 32]` | UNI spatial map used by SPADE at D5 |
| `uni_maps[64]` | `[B, 256, 64, 64]` | UNI spatial map used by SPADE at D4 |
| `uni_maps[128]` | `[B, 128, 128, 128]` | UNI spatial map used by SPADE at D3 |
| `uni_maps[256]` | `[B, 64, 256, 256]` | UNI spatial map used by SPADE at D2 |

In the current branch, `uni_features` is the input to the cross-attention module. The `uni_maps[...]` tensor family comes from the UNI spatial processor and feeds SPADE when `spade_use_uni=True`.

### 2.2 Decoder feature shapes in the default 512 model

| Stage | Feature shape after local conv | Cross-attention output | SPADE input | Final shape after SPADE and residual |
|---|---|---|---|---|
| D5 | `[B, 512, 32, 32]` | `[B, 512, 32, 32]` | `[B, 512, 32, 32]` | `[B, 512, 32, 32]` |
| D4 | `[B, 256, 64, 64]` | `[B, 256, 64, 64]` | `[B, 256, 64, 64]` | `[B, 256, 64, 64]` |
| D3 | `[B, 128, 128, 128]` | `[B, 128, 128, 128]` | `[B, 128, 128, 128]` | `[B, 128, 128, 128]` |
| D2 | `[B, 64, 256, 256]` | `[B, 64, 256, 256]` | `[B, 64, 256, 256]` | `[B, 64, 256, 256]` |

For the 1024 model, there is an extra D1 stage at 512 resolution. That stage uses an extra attention-to-SPADE projection, explained later.

## 3. Cross-attention layer: what it is and why it exists

The cross-attention layer is the semantic injection point.

### 3.1 Input and output contract

For one decoder stage, the layer receives:

- spatial features `x` with shape `[B, C, H, W]`,
- UNI tokens `uni_tokens` with shape `[B, N, 1024]`.

It returns a spatial tensor of the same shape as `x`:

- output shape `[B, C, H, W]`.

### 3.2 Why this layer exists

The decoder already has spatial information from the UNet path. What it lacks is pathology semantics.

Cross-attention lets every spatial location in the decoder ask:

- "Which UNI tokens are relevant to me?"

That is how global semantic context enters a local decoder feature map.

### 3.3 Layer-by-layer breakdown of `CrossAttention`

The implementation is in `src/models/blocks.py`.

#### 1. `GroupNorm(32, channels)`

- **Input:** `[B, C, H, W]`
- **Output:** `[B, C, H, W]`
- **Purpose:** stabilize the spatial feature map before query projection.
- GroupNorm is used instead of BatchNorm because the batch size is often small and the model benefits from per-sample normalization.

#### 2. `q_proj = Conv2d(channels, channels, 1)`

- **Input:** normalized spatial features `[B, C, H, W]`
- **Output:** query features `[B, C, H, W]`
- **Purpose:** convert each spatial position into a query vector.
- The `1x1` convolution mixes channels without changing spatial resolution.

#### 3. `LayerNorm(1024)` on UNI tokens

- **Input:** `[B, N, 1024]`
- **Output:** `[B, N, 1024]`
- **Purpose:** normalize the token embeddings so the key and value projections operate on a stable semantic scale.

#### 4. `k_proj = Linear(1024, C)`

- **Input:** `[B, N, 1024]`
- **Output:** `[B, N, C]`
- **Purpose:** convert each UNI token into a key vector that can be compared with the spatial queries.

#### 5. `v_proj = Linear(1024, C)`

- **Input:** `[B, N, 1024]`
- **Output:** `[B, N, C]`
- **Purpose:** convert each UNI token into a value vector carrying the information that may be written back into the decoder.

#### 6. Attention score computation

The module reshapes the tensors into multi-head form:

- queries: `[B, heads, HW, head_dim]`
- keys: `[B, heads, head_dim, N]`
- values: `[B, heads, N, head_dim]`

Then it computes:

$$
A = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)
$$

- **Input:** `Q` and `K`
- **Output:** attention weights `A` with shape `[B, heads, HW, N]`
- **Purpose:** decide which UNI tokens matter for each spatial location.

#### 7. Weighted sum with values

$$
O = AV
$$

- **Input:** attention weights `A` and values `V`
- **Output:** attended features reshaped back to `[B, C, H, W]`
- **Purpose:** collect the UNI semantic information that is relevant to each decoder position.

#### 8. `out_proj = Conv2d(channels, channels, 1)`

- **Input:** attended map `[B, C, H, W]`
- **Output:** refined attended map `[B, C, H, W]`
- **Purpose:** mix the attended channels back into the decoder channel space before the residual add.

#### 9. Residual output

The final output of cross-attention is:

$$
\text{x\_attn} = x + \mathrm{out\_proj}(O)
$$

- **Input:** the original local feature map `x`
- **Output:** the attention-enriched feature map `x_attn`
- **Purpose:** keep the original decoder representation and add a learned semantic correction instead of replacing the decoder features completely.

## 4. The attention residual path

This is the independent attention path the user asked about.

At every decoder stage the code does:

1. build the local decoder feature map `x_conv`,
2. compute `x_attn = CrossAttention(x_conv, uni_features)` when attention is enabled,
3. optionally add the attention correction independently:

$$
\text{attention correction} = x_{attn} - x_{conv}
$$

and then

$$
\text{x} \leftarrow \text{x} + (x_{attn} - x_{conv})
$$

### Why this matters

- `x_conv` is the local convolutional state.
- `x_attn` is the semantic version of that same state.
- `x_attn - x_conv` is the learned correction from UNI tokens.
- This lets the model keep the decoder’s spatial structure while adding semantic refinement.

### Interpretation

This residual path is independent from SPADE.

- If `enable_attention_residual=True`, the attention branch modifies the decoder even if SPADE is not using attention.
- If `use_attention_for_spade=True`, the same attention output can also be used as SPADE conditioning.
- The two uses are separate and can be enabled together or independently.

## 5. SPADE block: what it does and why it is there

The SPADE block is the modulation layer that sits after the decoder convolution at each stage.

### 5.1 Input and output contract

For one stage, SPADE receives:

- `x` with shape `[B, C, H, W]`
- `uni_spatial` with shape `[B, uni_ch, H, W]` or `None`
- `class_emb` with shape `[B, 64]`

It returns:

- `[B, C, H, W]`

### 5.2 Purpose of each layer inside `SPADEBlock`

#### 1. `InstanceNorm2d(norm_channels)`

- **Input:** `[B, C, H, W]`
- **Output:** `[B, C, H, W]`
- **Purpose:** remove per-instance contrast and mean shifts before modulation.
- This gives the modulation layers a cleaner base to control.

#### 2. `spade_shared = Conv2d(uni_channels, hidden, 3, padding=1) + LeakyReLU`

- **Input:** the SPADE conditioning map `uni_spatial`
- **Output:** hidden spatial feature map `[B, hidden, H, W]`
- **Purpose:** extract a compact conditioning representation from the UNI spatial map or from the attention output if that is being routed into SPADE.

#### 3. `spade_gamma = Conv2d(hidden, C, 3, padding=1)`

- **Input:** hidden conditioning map `[B, hidden, H, W]`
- **Output:** spatial scale map `gamma_s` with shape `[B, C, H, W]`
- **Purpose:** learn where to amplify or suppress each decoder channel spatially.

#### 4. `spade_beta = Conv2d(hidden, C, 3, padding=1)`

- **Input:** hidden conditioning map `[B, hidden, H, W]`
- **Output:** spatial bias map `beta_s` with shape `[B, C, H, W]`
- **Purpose:** learn spatial offsets for each decoder channel.

#### 5. `film_gamma = Linear(64, C)`

- **Input:** stain embedding `[B, 64]`
- **Output:** channel scale vector `[B, C]`, reshaped to `[B, C, 1, 1]`
- **Purpose:** inject stain identity globally, independent of spatial location.

#### 6. `film_beta = Linear(64, C)`

- **Input:** stain embedding `[B, 64]`
- **Output:** channel bias vector `[B, C]`, reshaped to `[B, C, 1, 1]`
- **Purpose:** give the model stain-specific offsets at the channel level.

### 5.3 SPADE output formula

The block computes:

$$
\mathrm{SPADE}(x) = (\gamma_s + \gamma_c) \odot \mathrm{IN}(x) + (\beta_s + \beta_c)
$$

Where:

- `IN(x)` is the instance-normalized feature map,
- `gamma_s` and `beta_s` come from the spatial conditioning map,
- `gamma_c` and `beta_c` come from the stain label embedding.

### 5.4 What happens if SPADE does not use UNI

The current code supports label-only SPADE.

If `uni_spatial=None`, then:

- `gamma_s = 0`
- `beta_s = 0`

So the block becomes label-only FiLM modulation:

$$
\mathrm{SPADE}(x) = \gamma_c \odot \mathrm{IN}(x) + \beta_c
$$

This is useful for ablation studies because it isolates the effect of the stain label from the effect of UNI spatial conditioning.

## 6. How cross-attention and SPADE connect in the decoder

This is the key point.

At each decoder stage the flow is:

1. upsample the current decoder state,
2. concatenate the decoder state with the encoder skip connection and optional edge features,
3. apply the local convolution for that stage,
4. optionally compute cross-attention,
5. feed either UNI spatial maps or attention output into SPADE,
6. optionally add the independent attention residual,
7. activate the result.

### Shared pattern for D5, D4, D3, and D2

For a generic decoder stage:

- `x_conv` = local fused decoder feature map,
- `x_attn` = cross-attention output from `x_conv` and `uni_features`,
- `spade_map` = `uni_maps[resolution]` or attention output,
- final stage output = SPADE output plus optional attention residual, then activation.

### Stage-by-stage purpose

#### D5 at 32×32

- **Local conv input:** upsampled bottleneck plus `e4` skip plus optional edge features.
- **Purpose of the local conv:** fuse spatial structure at the first decoder level.
- **Cross-attention purpose:** inject high-level semantic context into the 32×32 map.
- **SPADE purpose:** modulate the 32×32 feature map with UNI spatial conditioning and stain label.
- **Residual purpose:** add a direct attention correction without replacing the convolutional path.

#### D4 at 64×64

- **Local conv input:** upsampled D5 output plus `e3` skip plus optional edge features.
- **Purpose:** refine medium-scale spatial structure.
- **Cross-attention purpose:** re-inject UNI semantics after the resolution change.
- **SPADE purpose:** keep stain-aware modulation active at this new scale.
- **Residual purpose:** preserve the attention correction through the stage transition.

#### D3 at 128×128

- **Local conv input:** upsampled D4 output plus `e2` skip plus optional edge features.
- **Purpose:** refine finer anatomical structure.
- **Cross-attention purpose:** maintain semantic consistency as the map gets larger.
- **SPADE purpose:** modulate local structure at 128×128 with UNI-based conditioning.
- **Residual purpose:** keep the semantic correction independent of the SPADE branch.

#### D2 at 256×256

- **Local conv input:** upsampled D3 output plus `e1` skip plus optional edge features.
- **Purpose:** prepare the high-resolution representation that leads into the final output stage.
- **Cross-attention purpose:** keep the decoder semantically grounded before final refinement.
- **SPADE purpose:** adjust the 256×256 map using UNI spatial context and class label.
- **Residual purpose:** preserve the attention correction after the SPADE block.

## 7. The 1024 model only: the D1 attention-to-SPADE bridge

The 1024 version adds one more decoder stage at 512 resolution.

### Additional shape flow

- `dec1_conv` output: `[B, 64, 512, 512]`
- `dec1_attn` output: `[B, 64, 512, 512]`
- `dec1_attn_to_spade`: `[B, 64, 512, 512] -> [B, 32, 512, 512]`
- `dec1_spade` input: `[B, 64, 512, 512]` feature map plus `[B, 32, 512, 512]` conditioning map

### Why `dec1_attn_to_spade` exists

At this stage the SPADE block expects a 32-channel conditioning map, but the attention output has 64 channels.

So the `1x1` convolution acts as a channel bridge:

- it compresses the attention output from 64 channels to 32 channels,
- it keeps the spatial resolution unchanged,
- it makes the attention output compatible with the SPADE block.

This layer exists only in the 1024 branch.

## 8. What the toggles mean in practice

The current code lets you separate the attention uses.

| Toggle | Effect |
|---|---|
| `use_attention_for_spade=True` | SPADE receives the attention output instead of the normal UNI spatial map |
| `enable_attention_residual=True` | The attention delta is added back to the decoder independently of SPADE |
| `spade_use_uni=False` | SPADE becomes label-only FiLM and ignores UNI spatial conditioning |

### Important combined cases

- `use_attention_for_spade=False`, `enable_attention_residual=True`
  - attention acts as an independent residual correction,
  - SPADE still uses the normal UNI spatial maps.

- `use_attention_for_spade=True`, `enable_attention_residual=False`
  - attention mainly serves as SPADE conditioning,
  - the independent residual is off.

- Both enabled
  - attention influences the decoder twice in two different roles:
    - as SPADE conditioning,
    - as an independent additive correction.

- Both disabled
  - the cross-attention module is not used,
  - SPADE uses the normal UNI spatial maps only.

## 9. Short intuition

If you want a one-sentence summary:

- the convolution path builds the local image structure,
- cross-attention injects global UNI semantics,
- SPADE turns those semantics into spatially adaptive modulation,
- the residual path keeps the attention signal available as an explicit correction.

That is why the decoder has both a SPADE-conditioned path and an independent attention path.

## 10. Practical reading guide

If you want to inspect the implementation in code order, read it in this order:

1. [src/models/blocks.py](../src/models/blocks.py)
2. [src/models/generator.py](../src/models/generator.py)
3. [src/models/trainer.py](../src/models/trainer.py)

The most important lines are the `CrossAttention` class, the `SPADEBlock` class, and the decoder loop inside `SPADEUNetGenerator.forward`.
