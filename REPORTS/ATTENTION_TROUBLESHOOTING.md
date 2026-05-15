# UNIStainNet Attention Troubleshooting Note

This note answers three questions:

1. If the attention path is making all stains look alike, what is most likely wrong?
2. What is the bottleneck self-attention layer, when was it added, and what does it do?
3. What architecture changes should be tried next, and how should they be evaluated?

No code was changed for this note. I am only describing the most likely failure mode and the safest experiment plan.

---

## Short Answer

The most likely problem is not that Cross-Attention is mathematically broken. The stronger suspect is that the generator currently has a weak or dead stain-conditioning path, so the model can ignore the stain label and collapse toward a generic IHC look.

The biggest red flag in this repository is that the stain label embedding is computed in the generator and then discarded. In other words, the stain vector exists, but it is not actually used to modulate the output. That alone can make all stains look similar even if Cross-Attention itself is working.

The bottleneck self-attention is a separate global-context block at the 16×16 bottleneck. It mixes information across all spatial positions inside the compressed feature map. It is present in the current implementation, but it is not called out as a core component of the paper-facing architecture in this repo, so treat it as an implementation augmentation rather than the main paper mechanism.

---

## Most Likely Root Cause of “All Stains Look Alike”

### 1. The stain label path is effectively unused

In `SPADEUNetGenerator.forward`, the label embedding is computed, but the result is not used to condition any layer output. The code does this:

- `self.class_embed = nn.Embedding(num_classes, class_dim)` is created in the generator.
- In `forward`, the model calls `_ = self.class_embed(labels)` and then throws the result away.
- No FiLM, no SPADE-style modulation, no class-specific attention bias, and no class-conditioned decoder branch uses that vector.

That means the stain label does not actively separate HER2, Ki67, ER, and PR in the generator. If the user expects a “stain vector” to control stain appearance, it currently does not.

### Why this is important

If the only strong conditioning signal is the H-map plus UNI tokens, the network can learn to output a generic average IHC style. That often shows up as:

- HER2 missing membrane specificity.
- Ki67, ER, and PR converging toward the same brown nuclear pattern.
- Different stain labels producing almost identical outputs.

### 2. Cross-Attention may be too weak to carry stain identity by itself

Cross-Attention is semantically rich, but in this implementation it injects UNI tokens, not stain labels. UNI is useful for pathology context, but it is not a direct substitute for explicit stain conditioning.

If the attention output is small or diffuse, the residual path can make the block behave almost like identity. In that case the generator keeps relying on the U-Net skips and the H-map, and the stain-specific differences disappear.

### 3. Skip connections can overpower the semantic branch

The decoder still has very strong skip connections from the encoder. That is good for structure preservation, but it can also overpower the semantic signal from Cross-Attention if the attention branch is weak.

This is a common failure mode:

- Encoder/skip path carries the dominant structure.
- Attention path adds only a small correction.
- Output becomes stable but bland and stain-averaged.

### 4. UNI tokens may be too pooled or too generic

The trainer builds UNI features from sub-crops and then pools them into a token grid. If those tokens do not preserve enough stain-discriminative detail, Cross-Attention cannot recover it.

That does not mean Cross-Attention is wrong. It means the semantic source is too weak or too compressed.

### 5. Losses may encourage the average stain look

Some losses are alignment-free and broad. If the label-conditioning signal is weak, the optimizer may choose a low-risk solution that satisfies global color and DAB statistics but not stain-specific structure.

The result is often a visually plausible but stain-ambiguous generator.

---

## Bottleneck Self-Attention: What It Is

### Where it is in the network

The generator bottleneck is built as:

- `ResBlock(512)`
- `SelfAttention(512)`
- `ResBlock(512)`

This happens at the 16×16 latent feature map, right after the deepest encoder stage and before the decoder starts.

### What it does

The bottleneck self-attention is a global-context mixer inside the compressed feature map.

At a high level:

- It normalizes the 16×16 feature map.
- It builds Q, K, and V from the same tensor.
- Every spatial position can attend to every other spatial position.
- The output is added back to the input as a residual update.

### Why it exists

Its purpose is to let the model reason about long-range spatial relations at the most compressed stage. That can help when the network needs global shape consistency or global tissue organization.

### Why it can be neutral or harmful for stain separation

If the bottleneck attention learns broad, stain-agnostic context, it can smooth away the very differences you want the stain-specific decoder to preserve. It does not know anything about stain identity; it only mixes spatial context.

So if the model already struggles to separate stains, this block can be neutral or even slightly harmful if it becomes another source of averaging.

---

## When Was the Bottleneck Self-Attention Added?

Based on the repository history and current code, the bottleneck self-attention is present in the current implementation and is already part of the generator definition.

The changelog does not present it as a headline paper mechanism. The repo’s architecture evolution emphasizes the later move to Cross-Attention and the Eosin injection. The self-attention sits inside the bottleneck as an architectural augmentation, not as the main conditioning mechanism.

So the safest interpretation is:

- It is present in the current repo implementation.
- It appears to be a repo-level augmentation, not the central paper-defined conditioning path.
- I do not see evidence in this repository that it is the main original-paper feature.

If you want the strictest answer: treat it as implementation-specific, not as the core paper contribution.

---

## Is It Present in the Original Paper Implementation?

I would answer: not as a prominently documented core element in this repo’s paper-facing architecture.

The repository’s emphasis is:

- H-map encoder for structure.
- UNI Cross-Attention for semantics.
- Eosin injection for membrane topology.

The bottleneck self-attention is not the main conceptual contribution. It is a useful internal block, but it is not the feature the repo revolves around.

---

## What I Suggest Is Wrong, Ranked by Likelihood

### Most likely

1. The stain label embedding is not actually used to control the output.
2. Cross-Attention is too weak relative to the skip paths, so it cannot override the generic look.
3. UNI tokens are not sufficiently stain-discriminative after pooling.

### Less likely, but still possible

4. The bottleneck self-attention is smoothing features too much.
5. The loss mix is too forgiving and allows stain-averaged solutions.
6. The model is relying on structure-only cues and not enough stain-specific modulation.

---

## Architecture Edits I Would Try First

These are ordered from highest value to most invasive.

### 1. Actually use the stain embedding in the generator

This is the first thing I would fix.

Possible ways to do it:

- FiLM modulation at every decoder stage.
- Class-conditioned scale and bias on decoder activations.
- Add stain embedding into the Cross-Attention block as an extra conditioning vector.
- Use stain-specific adapter layers.

Why this should help:

- HER2, Ki67, ER, and PR do not just differ in structure. They differ in what kinds of staining should appear and where.
- A dead label embedding cannot enforce that.

### 2. Add a gate on the Cross-Attention output

Make the attention branch explicit instead of relying on an unconstrained residual add.

Example idea:

- `x = x + alpha * Attention(x, uni_tokens)`
- Initialize `alpha` small.
- Let it learn upward.

Why:

- If attention is too weak, the model ignores it.
- If it is too strong too early, training can become unstable.
- A learned gate makes the importance of attention visible.

### 3. Reduce skip-path dominance

Try one of the following:

- Add dropout on skip connections.
- Compress skip tensors with 1×1 convolutions before concatenation.
- Use attention only after a stronger decoder conv block.
- Add a stronger semantic branch at shallow decoder levels.

Why:

- The decoder may be reconstructing mainly from encoder skips.
- If so, the semantic path becomes decorative instead of controlling output.

### 4. Remove bottleneck self-attention in an ablation

This is a clean ablation, not necessarily a permanent fix.

Why:

- If the bottleneck self-attention is helping generic spatial mixing but not stain specificity, removing it may make the semantic branch easier to study.
- It also tells you whether the “everything looks alike” issue is coming from global bottleneck mixing rather than attention conditioning.

### 5. Increase the semantic specificity of UNI conditioning

Possible changes:

- Use higher-resolution UNI tokens.
- Use less aggressive pooling.
- Add stain-specific token projections.
- Feed UNI features into the decoder at more levels, not just through Cross-Attention.

Why:

- If UNI features become too homogeneous, the attention block has little to differentiate.

### 6. Add stain-aware supervision to the attention branch

Examples:

- Per-stain auxiliary classifier on generated output.
- Stain-consistency regularizer.
- Different decoder heads per stain group.

Why:

- If the model can get away with a generic output, it will.
- Explicit stain-specific supervision makes collapse less likely.

---

## Best Debugging Ablation Sequence

If I were running this project, I would test the following in order:

### Experiment 1: make the label embedding actually condition the decoder

Goal: check whether the stain collapse is mostly because the label path is dead.

What to compare:

- Before vs after stain separation in generated HER2 / Ki67 / ER / PR.
- W&B metrics: `val/dab_mae`, `val/lpips`, `val/ssim`.
- Per-stain qualitative grids.

Expected outcome if this is the real problem:

- Stains start diverging visibly.
- HER2 membrane staining becomes more distinct.
- Nuclear stains stop looking like the same generic brown output.

### Experiment 2: gate the Cross-Attention output

Goal: test whether attention is being ignored.

Expected outcome:

- If attention was too weak, gating should make stain-driven differences easier to learn.

### Experiment 3: remove bottleneck self-attention

Goal: check whether the bottleneck block is smoothing away stain-specific information.

Expected outcome:

- If outputs become more differentiated without the bottleneck self-attention, the block was hurting the conditioning path.

### Experiment 4: keep only one semantic path at a time

Test separately:

- Cross-Attention only.
- Label conditioning only.
- Cross-Attention plus label conditioning.

Expected outcome:

- This reveals which path actually carries stain identity.
- It also shows whether Cross-Attention is useful or just redundant.

---

## What I Would Measure

To know whether the problem is solved, I would look at:

- Visual separation between HER2, Ki67, ER, and PR.
- `val/dab_mae` per stain class.
- `val/lpips` and `val/ssim`.
- Class-wise DAB statistics and sample grids.
- Whether HER2 membrane staining becomes visibly different from nuclear stains.

A model can still score well on generic metrics and be wrong on stain identity, so visual inspection matters here.

---

## Important Constraint

You asked me to run epochs after the edits, but you also asked me not to edit any files in this step.

Because of that, I cannot actually run a modified experiment from here. I can only provide the recommendation set and the validation plan.

If you want, the next practical step would be to let me implement one of the changes above, then I can run a short ablation and compare the outputs.

---

## Bottom Line

If all stains look alike, the first thing I would blame is not Cross-Attention itself. I would blame the fact that the stain label embedding is not actually being used to modulate the generator.

The bottleneck self-attention is a secondary factor: it adds global spatial context at 16×16, but it is not a stain-conditioning mechanism. It was present by the V2-era bottleneck design in this repo, and it should be treated as an implementation augmentation rather than the original paper’s core idea.

The highest-value fix is to make the stain vector real, visible, and mandatory in the decoder path.
