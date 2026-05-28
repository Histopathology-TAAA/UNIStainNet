#!/bin/bash
# =============================================================================
# UNIStainNet Destaining Training — Lightning AI Launch Script
#
# Assumed Lightning AI layout:
#   /teamspace/studios/this_studio/     ($HOME on Lightning AI)
#       UNISTAINNET/          ← this repo
#       Destained_Results/    ← zipped datasets (HER2-Destained.zip, etc.)
#       data/MIST/            ← created by this script
#
# Run from: /teamspace/studios/this_studio/UNISTAINNET/
# NOTE: On Lightning AI, $HOME = /teamspace/studios/this_studio (not ~/teamspace/...)
# =============================================================================

set -e   # stop immediately on any error

STUDIO_ROOT="$HOME"  # on Lightning AI this is /teamspace/studios/this_studio
REPO_DIR="$STUDIO_ROOT/UNIStainNet"
ZIP_DIR="$STUDIO_ROOT/Destained_Results"
DATA_DIR="$STUDIO_ROOT/data/MIST"

# echo "========================================================"
# echo "  Step 0: Install the dataset from Hugging face"
# echo "========================================================"

# mkdir $ZIP_DIR
# cd $ZIP_DIR
# hf download asserelzeki/destained-histopathology-data --repo-type dataset --local-dir .
# cd $REPO_DIR

echo "========================================================"
echo "  Step 1: Unzip and rename stain folders"
echo "========================================================"

mkdir -p "$DATA_DIR"

for STAIN in HER2 ER Ki67 PR; do
    ZIP_PATH="$ZIP_DIR/${STAIN}-Destained.zip"
    TARGET_DIR="$DATA_DIR/$STAIN"

    if [ -d "$TARGET_DIR" ]; then
        echo "  [SKIP] $STAIN already exists at $TARGET_DIR"
        continue
    fi

    if [ ! -f "$ZIP_PATH" ]; then
        echo "  [ERROR] Zip not found: $ZIP_PATH"
        exit 1
    fi

    echo "  Unzipping $STAIN ..."
    unzip -q "$ZIP_PATH" -d "$DATA_DIR/"

    # Rename from *-Destained to plain stain name (required by STAIN_TO_LABEL)
    if [ -d "$DATA_DIR/${STAIN}-Destained" ]; then
        mv "$DATA_DIR/${STAIN}-Destained" "$TARGET_DIR"
        echo "  Renamed ${STAIN}-Destained → $STAIN"
    fi

    echo "  Done: $STAIN"
done

echo ""
echo "========================================================"
echo "  Step 2: Verify required subfolders exist"
echo "========================================================"

for STAIN in HER2 ER Ki67 PR; do
    for SPLIT in trainA trainA-H trainB trainB-H valA valA-H valB valB-H; do
        DIR="$DATA_DIR/$STAIN/$SPLIT"
        if [ ! -d "$DIR" ]; then
            echo "  [MISSING] $DIR"
            echo "  Fix: check the zip contents — folder name may differ."
            exit 1
        fi
    done
    echo "  [OK] $STAIN — all 8 subfolders present"
done

# echo ""
# echo "========================================================"
# echo "  Step 3: Install dependencies"
# echo "========================================================"

# cd "$REPO_DIR"
# pip install -r requirements.txt -q
# pip install -e . -q
# echo "  Dependencies installed."

# echo ""
# echo "========================================================"
# echo "  Step 4: CPU sanity check (no GPU, no UNI download)"
# echo "========================================================"

# export PYTHONPATH="$REPO_DIR"
# python scripts/debug/sanity_check_mist_dummy.py
# echo "  Architecture sanity check passed."

# echo ""
# echo "========================================================"
# echo "  Step 5: Real data loading check (CPU, no model)"
# echo "========================================================"

# python scripts/debug/validate_data.py \
#     --data_dir "$DATA_DIR" \
#     --stains HER2 ER Ki67 PR \
#     --batch_size 4 \
#     --n_batches 3
# echo "  Data loading check passed."

echo ""
echo "========================================================"
echo "  Step 6: Train  (GPU required from this point)"
echo "========================================================"
# Adjust --batch_size based on GPU:
#   A10G  24GB  → 4–6
#   A100  40GB  → 8       ← recommended starting point
#   A100  80GB  → 12–16

# python scripts/train/train_mist.py \
#     --data_dir   "$DATA_DIR" \
#     --stains     HER2 ER Ki67 PR \
#     --batch_size 8 \
#     --max_epochs 100 \
#     --ckpt_dir   "$STUDIO_ROOT/checkpoints/destaining_v1" \
#     --wandb_name destaining_v1_attention_batch8

# For copy and paste command running uncomment, copy, and use this code 

# python scripts/train/train_mist.py \
#     --data_dir   "$HOME/data/MIST" \
#     --stains     HER2 ER Ki67 PR \
#     --batch_size 8 \
#     --max_epochs 100 \
#     --ckpt_dir   "$HOME/checkpoints/destaining_v2" \
#     --wandb_name destaining_v2_attention_batch8


# change the wandb name if needed to log 

# to continue training on a checkpoint use this script

python scripts/train/train_mist.py \
    --data_dir   "$HOME/data/MIST" \
    --stains     HER2 ER Ki67 PR \
    --batch_size 8 \
    --max_epochs 100 \
    --ckpt_dir   "$HOME/checkpoints/destaining_v2" \
    --wandb_name destaining_v2_attention_batch8 \
    --resume_from "$HOME/checkpoints/destaining_v2/last.ckpt"

####################################
#########   Ablation Study #########
####################################

# Run 1 Attention Residual with normal SPADE

PYTHONPATH=. python scripts/train/train_mist.py \    
    --data_dir   "/home/ahmed_ayman/data/Destained_MIST" \
    --stains     ER PR \
    --batch_size 2      \
    --accum_steps 4      \
    --max_epochs 100      \
    --ckpt_dir   "./checkpoints/destaining_v2"     \
    --wandb_name destaining_v2_attention_batch4_ER_PR_Only \
    --use_eosin_encoder 


# for evaluation 
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "/home/ahmed_ayman/data/Models Checkpoints/UNIStainNet/Destaining_v2_ER_PR/attention spade only /mist_epoch=085_step=354557.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --batch_size 4 \
  --output_dir "./eval_output/destaining_v2_ER_PR_attn_spade_only"

PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v2/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --batch_size 4 \
  --output_dir "./eval_output/destaining_v2_ER_PR_residual_and_normal_spade"

# Run 1 Done 


# Run 2 SPADE attention block only
# Ongoing on Knights machine
source .venv/bin/activate
PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
    --stains ER PR \
    --batch_size 4 \
    --accum_steps 2 \
    --ckpt_dir "checkpoints/destaining_v2_attn_spade" \
    --edge_encoder v2 \
    --disable_attention_residual \
    --use_eosin_encoder \
    --use_attention_for_spade   \
    --wandb_name destaining_v2_attention_batch4_ER_PR_Only_attn_spade_Only_fixed 

# Run 2 Done

# Run 3 both attention paths
# Ongoing on Knights
PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
    --stains ER PR \
    --batch_size 4 \
    --accum_steps 2 \
    --ckpt_dir "checkpoints/destaining_v2_attn_spade_and_residual" \
    --edge_encoder v2 \
    --enable_attention_residual \
    --use_eosin_encoder \
    --use_attention_for_spade   \
    --wandb_name destaining_v2_attention_batch4_ER_PR_Only_attn_spade_and_residual_together_fixed \
    --log_val_fid \
    --use_h_adapter 

PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v2_attn_spade_and_residual/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --batch_size 4 \
  --output_dir "./eval_output/destaining_v2_attn_spade_and_residual"
# Done


# Run 4 Residual with no SPADE
PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
    --stains ER PR \
    --batch_size 4 \
    --accum_steps 2 \
    --ckpt_dir "checkpoints/destaining_v2_attn_residual_only" \
    --edge_encoder v2 \
    --enable_attention_residual \
    --use_eosin_encoder \
    --no_spade_use_uni \
    --no_use_attention_for_spade  \
    --wandb_name destaining_v2_attention_batch4_ER_PR_Only_residual_Only

# Run 5 Normal SPADE
# On Knights Machine
PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
    --stains ER PR \
    --batch_size 4 \
    --accum_steps 2 \
    --ckpt_dir "checkpoints/destaining_v2_attn_Normal_SPADE" \
    --edge_encoder v2 \
    --disable_attention_residual \
    --use_eosin_encoder \
    --no_use_attention_for_spade  \
    --wandb_name destaining_v2_attention_batch4_ER_PR_Only_Normal_SPADE \
    --log_val_fid \
    --use_h_adapter
# Done
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v2_attn_Normal_SPADE/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --batch_size 4 \
  --output_dir "./eval_output/destaining_v2_Normal_SPADE"


####################################
#########  New Ideas Runs  #########
####################################

# Run 6 100% H&E H input with normal spade with no attention
# 100% H&E H input with normal spade with no attention
# Ongoing on Knights machine
PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
    --stains ER PR \
    --batch_size 16 \
    --accum_steps 2 \
    --ckpt_dir "checkpoints/destaining_v2_HE_only_Normal_SPADE" \
    --edge_encoder v2 \
    --disable_attention_residual \
    --use_eosin_encoder \
    --no_use_attention_for_spade  \
    --wandb_name destaining_v2_attention_batch16_ER_PR_Only_HE_only_Normal_SPADE_fixed \
    --log_val_fid \
    --resume_from "checkpoints/destaining_v2_HE_only_Normal_SPADE/last.ckpt"

PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v2_HE_only_Normal_SPADE/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --batch_size 4 \
  --output_dir "./eval_output/destaining_v2_HE_only_Normal_SPADE"
  
# Done

# Run 7 same as run 2 but change losses weights which are
# 'lpips_weight': 1.0,
# 'lpips_256_weight': 0.5,
# 'lpips_512_weight': 0.0,
# 'l1_fullres_weight': 1.5,           # ↑ from 1.0 — more pixel pressure → better SSIM
# 'lpips_fullres_weight': 1.0,
# 'he_edge_weight': 0.5,
# 'l1_lowres_weight': 1.0,
# 'adversarial_weight': 0.0,
# 'uncond_disc_weight': 2.0,          # ↑ from 1.0 — stronger unconditional realism → better FID
# 'dab_intensity_weight': 0.2,
# 'dab_contrast_weight': 0.2,         # ↑ from 0.0 — enforces class ordering → better DAB KL/JSD
# 'dab_sharpness_weight': 0.0,
# 'gram_style_weight': 0.5,           # ↑ from 0.0 — Gram texture matching → better FID
# 'edge_weight': 0.0,
# 'crop_disc_weight': 0.0,
# 'feat_match_weight': 10.0,
# 'patchnce_weight': 0.0,
# 'bg_white_weight': 0.0,
# # IHC / DAB losses
# 'ihc_edge_weight': 0.1,
# 'dab_histo_weight': 0.5,            # ↑ from 0.3 — Wasserstein-1 on OD distribution → better DAB KL/JSD
# 'dab_block_weight': 0.5,
# 'dab_block_size': 32,
# 'dab_sparsity_weight': 0.3,
# 'dab_sparsity_margin': 0.05,
# # Stain-conditioned discriminator
# 'proj_disc_weight': 3.0,            # ↑ from 2.0 — stain-conditioned realism → better FID
# Ongoing on Knights machine
PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
    --stains ER PR \
    --batch_size 4 \
    --ckpt_dir "checkpoints/destaining_v2_attn_spade_new_weights" \
    --edge_encoder v2 \
    --disable_attention_residual \
    --use_eosin_encoder \
    --use_attention_for_spade   \
    --wandb_name destaining_v2_attention_batch4_ER_PR_Only_attn_spade_Only_new_weights \
    --log_val_fid






# To upload a model to hf
# hf upload asserelzeki/destained_v1_UNIStainnet_v2 ./checkpoints/destaining_v2/mist_epoch=002_step=010973.ckpt
# hf upload asserelzeki/destained_UNIStainnet_v2 ./checkpoints/destaining_v2/mist_epoch=002_step=010973.ckpt
# hf upload asserelzeki/destained_UNIStainnet_v2_attn_SPADE_only ./checkpoints/destaining_v2_attn_spade/last.ckpt destained_UNIStainnet_v2_attn_SPADE_only.ckpt