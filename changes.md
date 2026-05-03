# Refactor Changelog (Structure-Semantic Decoupled + Mixed-Domain)

## Dataset
- Added paired loading of RGB images and 1-channel H-maps per stain with the new folder layout: trainA, trainA-H, trainB, trainB-H, valA, valA-H, valB, valB-H.
- Ensured paired crops and spatial augmentations are applied identically across H&E RGB, IHC RGB, H&E H-map, and IHC H-map.
- Normalized H-maps as 1-channel tensors using mean=[0.5], std=[0.5].

## Architecture
- Decoupled structure and semantics: the UNet encoder now consumes 1-channel structural inputs (H-maps) instead of RGB.
- Replaced SPADE conditioning blocks with 4-head Cross-Attention blocks that use GroupNorm; Q from spatial features, K/V from UNI tokens.
- Added residual connections around Cross-Attention to avoid bottlenecks.

## Training Logic
- Implemented 50/50 mixed-domain routing per batch:
  - Case A (aligned): generator input is IHC H-map; losses are full-res L1 and full-res LPIPS against IHC RGB.
  - Case B (misaligned): generator input is H&E H-map; losses are L1 at 64x64 and LPIPS at 128 and 256.
- UNI semantic tokens are always extracted from H&E RGB.
- Validation uses H&E H-map for generation and evaluates against IHC RGB.

## Discriminator
- Kept the discriminator unconditional by evaluating only IHC images (real vs generated), not concatenated with H&E inputs.
- Feature matching and R1 penalty remain computed from unconditional discriminator features.

## Sanity Check Script
- Dummy-data sanity check lives only in scripts/debug/sanity_check_mist_dummy.py and does not affect training code.
- It bypasses UNI extraction internally to avoid model downloads, and runs one train and one validation step.

## How To Run
- From repo root:
  - PowerShell:
    - Set-Location .
    - $env:PYTHONPATH = "$PWD"
    - .\.venv\Scripts\python.exe scripts\debug\sanity_check_mist_dummy.py
