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
    --batch_size 8 \
    --ckpt_dir "checkpoints/baseline_ER_PR_only" \
    --wandb_name baseline_batch8_ER_PR_Only 

PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/baseline_ER_PR_only/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --batch_size 4 \
  --output_dir "./eval_output/baseline_ER_PR_only"

# Baseline small addition runs
PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
    --stains ER PR \
    --case_a_prob 0 \
    --batch_size 8 \
    --ckpt_dir "checkpoints/destaining_v3_he_only" \
    --wandb_name destaining_v3_batch8_he_only \
    --resume_from "./checkpoints/destaining_v3_he_only/last.ckpt"

PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v3_he_only/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --output_dir "./eval_output/destaining_v3_he_only"

# Done

# Run 8
PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
    --stains ER PR \
    --case_a_prob 0.5 \
    --batch_size 8 \
    --edge_encoder v2 \
    --ckpt_dir "checkpoints/destaining_v3_0.5_ihc_only" \
    --wandb_name destaining_v3_batch8_0.5_ihc_only \
    --use_alignment 
    # --resume_from "./checkpoints/destaining_v3_0.5_ihc_only/last.ckpt" 

PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v3_0.5_ihc_only/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --output_dir "./eval_output/destaining_v3_0.5_ihc_only_aligned" \
  --enable_stn_alignment \
  --aligned 


PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
    --stains ER PR \
    --case_a_prob 0.75 \
    --batch_size 8 \
    --edge_encoder v2 \
    --ckpt_dir "checkpoints/destaining_v3_0.75_ihc_only" \
    --wandb_name destaining_v3_batch8_0.75_ihc_only \
    --use_alignment 
    --resume_from "./checkpoints/destaining_v3_0.5_ihc_only/last.ckpt"

PYTHONPATH=. python scripts/eval/eval_mist.py \
    --checkpoint "./checkpoints/destaining_v3_0.75_ihc_only/last.ckpt" \
    --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
    --stains ER PR \
    --output_dir "./eval_output/destaining_v3_0.75_ihc_only" \
    --enable_stn_alignment \
    --aligned


PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
    --stains ER PR \
    --batch_size 8 \
    --edge_encoder v2 \
    --case_a_prob 0.5 \
    --uni_perceptual_weight 1.0 \
    --use_alignment \
    --ckpt_dir "checkpoints/destaining_v3_batch8_0.5_UNI_loss_ihc_only" \
    --wandb_name destaining_v3_batch8_0.5_UNI_loss_ihc_only \
    --resume_from "./checkpoints/destaining_v3_batch8_0.5_UNI_loss_ihc_only/last.ckpt" 

PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v3_batch8_0.5_UNI_loss_ihc_only/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --output_dir "./eval_output/destaining_v3_batch8_0.5_UNI_loss_ihc_only_aligned" \
  --enable_stn_alignment \
  --aligned 
PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
    --stains ER PR \
    --batch_size 8 \
    --edge_encoder v2 \
    --case_a_prob 0.5 \
    --learnable_sobel \
    --use_alignment \
    --ckpt_dir "checkpoints/destaining_v3_batch8_0.5_learnable_sobel" \
    --wandb_name destaining_v3_batch8_0.5_learnable_sobel 
    # --resume_from "./checkpoints/destaining_v3_batch8_0.5_learnable_sobel/last.ckpt"

PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v3_batch8_0.5_learnable_sobel/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --output_dir "./eval_output/destaining_v3_batch8_0.5_learnable_sobel_aligned" \
  --enable_stn_alignment \
  --aligned 
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v3_batch8_0.5_learnable_sobel/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER \
  --output_dir "./eval_output/destaining_v3_batch8_0.5_learnable_sobel_aligned" \
  --enable_stn_alignment \
  --aligned 

PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
    --stains ER PR \
    --batch_size 8 \
    --edge_encoder v2 \
    --case_a_prob 0.5 \
    --kstain_perceptual_weight 1.0 \
    --use_alignment \
    --ckpt_dir "checkpoints/destaining_v3_batch8_0.5_K_Stain_Run_1" \
    --wandb_name destaining_v3_batch8_0.5_K_Stain_Run_1 \
    --resume_from "./checkpoints/destaining_v3_batch8_0.5_K_Stain_Run_1/last.ckpt"

PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v3_batch8_0.5_K_Stain_Run_1/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --output_dir "./eval_output/destaining_v3_batch8_0.5_K_Stain_Run_1_aligned" \
  --enable_stn_alignment \
  --aligned 

PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
    --stains ER PR \
    --batch_size 8 \
    --edge_encoder v2 \
    --case_a_prob 0.5 \
    --kstain_perceptual_weight 1.0 \
    --use_alignment \
    --ckpt_dir "checkpoints/destaining_v3_batch8_0.5_K_Stain_wasserstein_loss_Run_2" \
    --wandb_name destaining_v3_batch8_0.5_K_Stain_Run_1 \
    --max_epochs 150 \
    --resume_from "./checkpoints/destaining_v3_batch8_0.5_K_Stain_Run_1/last.ckpt"  

PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v3_batch8_0.5_K_Stain_wasserstein_loss_Run_2/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER \
  --output_dir "./eval_output/destaining_v3_batch8_0.5_K_Stain_Run_2_aligned" \
  --enable_stn_alignment \
  --aligned

# --enable_stn_alignment
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

PYTHONPATH=. python scripts/train/train_mist.py     --data_dir "/home/ahmed_ayman/data/Destained_MIST"     --stains ER PR     --batch_size 8     --edge_encoder v2     --case_a_prob 0.5     --learnable_sobel     --he_rgb_dropout 0.0     --ckpt_dir "checkpoints/destaining_v3_learnable_sobel_no_dropout"     --wandb_name destaining_v3_learnable_sobel_no_dropout     --max_epochs 100

PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v3_learnable_sobel_no_dropout/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --output_dir "./eval_output/destaining_v3_learnable_sobel_no_dropout_aligned" \
  --enable_stn_alignment \
  --aligned
#   Done
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v3_learnable_sobel_no_dropout/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --output_dir "./eval_output/destaining_v3_learnable_sobel_no_dropout" \
  --enable_stn_alignment

PYTHONPATH=. python scripts/train/train_mist.py     --data_dir "/home/ahmed_ayman/data/Destained_MIST"     --stains ER PR     --batch_size 8     --edge_encoder v2     --case_a_prob 0.5     --learnable_sobel     --he_rgb_dropout 0.2     --ckpt_dir "checkpoints/destaining_v3_learnable_sobel_dropout_0.2"     --wandb_name destaining_v3_learnable_sobel_dropout_0.2     --max_epochs 100
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v3_learnable_sobel_dropout_0.2/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --output_dir "./eval_output/destaining_v3_learnable_sobel_dropout_0.2_aligned" \
  --enable_stn_alignment \
  --aligned
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v3_learnable_sobel_dropout_0.2/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --output_dir "./eval_output/destaining_v3_learnable_sobel_dropout_0.2" \
  --enable_stn_alignment


PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
    --stains ER PR \
    --batch_size 8 \
    --edge_encoder v2 \
    --case_a_prob 0.5 \
    --learnable_sobel \
    --he_rgb_dropout 0.2 \
    --ihc_augmentation 0.8 \
    --ckpt_dir "checkpoints/destaining_v3_anti_cheat_dropout_0.2" \
    --wandb_name destaining_v3_anti_cheat_dropout_0.2 \
    --max_epochs 100
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v3_anti_cheat_dropout_0.2/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --output_dir "./eval_output/destaining_v3_anti_cheat_dropout_0.2_aligned" \
  --enable_stn_alignment \
  --aligned
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v3_anti_cheat_dropout_0.2/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --output_dir "./eval_output/destaining_v3_anti_cheat_dropout_0.2" \
  --enable_stn_alignment   

PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
    --stains ER PR \
    --batch_size 8 \
    --edge_encoder v2 \
    --case_a_prob 0.5 \
    --learnable_sobel \
    --he_rgb_dropout 0.2 \
    --ihc_augmentation 0.8 \
    --ckpt_dir "checkpoints/destaining_v3_anti_cheat_dropout_0.2_with_alignment" \
    --wandb_name destaining_v3_anti_cheat_dropout_0.2_with_alignment \
    --use_alignment \
    --max_epochs 100
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v3_anti_cheat_dropout_0.2_with_alignment/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --output_dir "./eval_output/destaining_v3_anti_cheat_dropout_0.2_with_alignment_aligned" \
  --enable_stn_alignment \
  --aligned
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v3_anti_cheat_dropout_0.2_with_alignment/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --output_dir "./eval_output/destaining_v3_anti_cheat_dropout_0.2_with_alignment" \
  --enable_stn_alignment

PYTHONPATH=. python scripts/train/train_mist.py     --data_dir "/home/ahmed_ayman/data/Destained_MIST"     --stains ER PR     --batch_size 8     --edge_encoder v2     --case_a_prob 0.5     --learnable_sobel     --he_rgb_dropout 0.2     --bilateral 0.8     --use_alignment     --ckpt_dir "checkpoints/destaining_v3_bilateral_0.8_with_alignment"     --wandb_name destaining_v3_bilateral_0.8_with_alignment     --max_epochs 100
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v3_bilateral_0.8_with_alignment/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --output_dir "./eval_output/destaining_v3_bilateral_0.8_with_alignment_aligned" \
  --enable_stn_alignment \
  --aligned
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v3_bilateral_0.8_with_alignment/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --output_dir "./eval_output/destaining_v3_bilateral_0.8_with_alignment" \
  --enable_stn_alignment
PYTHONPATH=. python scripts/train/train_mist.py     --data_dir "/home/ahmed_ayman/data/Destained_MIST"     --stains ER PR     --batch_size 8     --edge_encoder v2     --case_a_prob 0.5     --learnable_sobel     --he_rgb_dropout 0.2     --bilateral 0.8     --ckpt_dir "checkpoints/destaining_v3_bilateral_0.8_no_alignment"     --wandb_name destaining_v3_bilateral_0.8_no_alignment     --max_epochs 100
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v3_bilateral_0.8_no_alignment/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --output_dir "./eval_output/destaining_v3_bilateral_0.8_no_alignment_aligned" \
  --enable_stn_alignment \
    --aligned
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/destaining_v3_bilateral_0.8_no_alignment/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains ER PR \
  --output_dir "./eval_output/destaining_v3_bilateral_0.8_no_alignment_aligned" \
  --enable_stn_alignment

PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
    --stains Ki67 \
    --batch_size 8 \
    --edge_encoder v2 \
    --case_a_prob 0 \
    --learnable_sobel \
    --he_rgb_dropout 0.0 \
    --bilateral 0.0 \
    --ckpt_dir "checkpoints/ki67_baseline_cheating_allowed" \
    --wandb_name ki67_baseline_cheating_allowed \
    --max_epochs 150 \
    --use_alignment \ 
    --resume_from "./checkpoints/ki67_baseline_cheating_allowed/last.ckpt"
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/ki67_baseline_cheating_allowed/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains Ki67 \
  --output_dir "./eval_output/ki67_baseline_cheating_allowed_aligned" \
  --enable_stn_alignment \
  --aligned
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/ki67_baseline_cheating_allowed/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains Ki67 \
  --output_dir "./eval_output/ki67_baseline_cheating_allowed" \
  --enable_stn_alignment

PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
    --stains Ki67 \
    --batch_size 8 \
    --edge_encoder v2 \
    --case_a_prob 0.5 \
    --learnable_sobel \
    --he_rgb_dropout 0.2 \
    --bilateral 0.8 \
    --ckpt_dir "checkpoints/ki67_anti_cheat_bilateral_0.8" \
    --wandb_name ki67_anti_cheat_bilateral_0.8 \
    --max_epochs 100 \
    --use_alignment
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/ki67_anti_cheat_bilateral_0.8/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains Ki67 \
  --output_dir "./eval_output/ki67_anti_cheat_bilateral_0.8_aligned" \
  --enable_stn_alignment \
  --aligned
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/ki67_anti_cheat_bilateral_0.8/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains Ki67 \
  --output_dir "./eval_output/ki67_anti_cheat_bilateral_0.8" \
  --enable_stn_alignment

# To upload a model to hf
# hf upload asserelzeki/destained_v1_UNIStainnet_v2 ./checkpoints/destaining_v2/mist_epoch=002_step=010973.ckpt
# hf upload asserelzeki/destained_UNIStainnet_v2 ./checkpoints/destaining_v2/mist_epoch=002_step=010973.ckpt
# hf upload asserelzeki/destained_UNIStainnet_v2_attn_SPADE_only ./checkpoints/destaining_v2_attn_spade/last.ckpt destained_UNIStainnet_v2_attn_SPADE_only.ckpt

PYTHONPATH=.python scripts/eval/metric_stress_test.py --image_path  "/home/ahmed_ayman/data/Destained_MIST/ER/valB/2M2103108_14_15.jpg" 

mkdir -p "deepliif-weights" && wget -qO- "https://zenodo.org/record/4751737/files/DeepLIIF_Latest_Model.zip" > "deepliif-weights/DeepLIIF_Latest_Model.zip" && cd "deepliif-weights" && unzip -q "DeepLIIF_Latest_Model.zip" && rm "DeepLIIF_Latest_Model.zip" && ls -la "DeepLIIF_Latest_Model"



PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir /home/ahmed_ayman/data/Destained_MIST \
    --ckpt_dir "./checkpoints/ki67_deepliif_run1" \
    --stains Ki67 \
    --deepliif_weights_path deepliif-weights/DeepLIIF_Latest_Model/latest_net_G1.pth \
    --case_a_prob 0.5 \
    --use_alignment \
    --batch_size 8 \
    --wandb_name "ki67_deepliif_run1" \
    --resume_from "./checkpoints/ki67_deepliif_run1/last.ckpt" 
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/ki67_deepliif_run1/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains Ki67 \
  --output_dir "./eval_output/ki67_deepliif_run1_aligned" \
  --enable_stn_alignment \
  --aligned
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/ki67_deepliif_run1/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains Ki67 \
  --output_dir "./eval_output/ki67_deepliif_run1" \
  --enable_stn_alignment



PYTHONPATH=. python scripts/train/train_mist.py     --data_dir /home/ahmed_ayman/data/Destained_MIST     --ckpt_dir "./checkpoints/ki67_deepliif_run2"     --stains Ki67     --deepliif_weights_path deepliif-weights/DeepLIIF_Latest_Model/latest_net_G1.pth     --case_a_prob 0.5     --use_alignment     --batch_size 8     --wandb_name "ki67_deepliif_run2"

--bilateral 0.5 \
--ihc_augmentation 0.5 \


PYTHONPATH=. python scripts/train/train_mist.py     --data_dir /home/ahmed_ayman/data/Destained_MIST     --ckpt_dir "./checkpoints/ki67_deepliif_run2/"     --stains Ki67     --deepliif_weights_path deepliif-weights/DeepLIIF_Latest_Model/latest_net_G1.pth     --case_a_prob 0    --use_alignment     --batch_size 8     --wandb_name "ki67_deepliif_run2" --resume_from "./checkpoints/ki67_deepliif_run2/last.ckpt" --max_epochs 150
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/ki67_deepliif_run2/last-v1.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains Ki67 \
  --output_dir "./eval_output/ki67_deepliif_run2_aligned" \
  --enable_stn_alignment \
  --aligned
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/ki67_deepliif_run2/last-v1.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains Ki67 \
  --output_dir "./eval_output/ki67_deepliif_run2" \
  --enable_stn_alignment

PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir /home/ahmed_ayman/data/Destained_MIST \
    --ckpt_dir "./checkpoints/ki67_deepliif_run3" \
    --stains Ki67 \
    --deepliif_weights_path deepliif-weights/DeepLIIF_Latest_Model/latest_net_G1.pth \
    --case_a_prob 1.0 \
    --case_a_warmup_epochs 40 \
    --case_a_anneal_epochs 80 \
    --case_a_end_prob 0.05 \
    --use_alignment \
    --batch_size 8 \
    --wandb_name "ki67_deepliif_run3" \
    --max_epochs 150 
    
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/ki67_deepliif_run3/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains Ki67 \
  --output_dir "./eval_output/ki67_deepliif_run3_aligned" \
  --enable_stn_alignment \
  --aligned
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/ki67_deepliif_run3/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains Ki67 \
  --output_dir "./eval_output/ki67_deepliif_run3" \
  --enable_stn_alignment


PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir /home/ahmed_ayman/data/Destained_MIST \
    --stains Ki67 \
    --batch_size 8 \
    --wandb_name "baseline_ki67_only" \
    --max_epochs 150 \
    --resume_from "./checkpoints/baseline_ki67/last.ckpt" 
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/baseline_ki67/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains Ki67 \
  --output_dir "./eval_output/baseline_ki67_only" 


PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir /home/ahmed_ayman/data/Destained_MIST \
    --ckpt_dir "./checkpoints/ki67_deepliif_run4" \
    --stains Ki67 \
    --deepliif_weights_path deepliif-weights/DeepLIIF_Latest_Model/latest_net_G1.pth \
    --case_a_prob 1.0 \
    --case_a_warmup_epochs 40 \
    --case_a_anneal_epochs 80 \
    --case_a_end_prob 0 \
    --use_alignment \
    --batch_size 8 \
    --wandb_name "ki67_deepliif_run4" \
    --max_epochs 150 


hf upload asserelzeki/destained_v4_UNIStainnet ./checkpoints/ki67_deepliif_run3/last.ckpt run3_150epoch.ckpt
hf upload asserelzeki/destained_v4_UNIStainnet ./checkpoints/ki67_deepliif_run2/last.ckpt run2_150epoch.ckpt
hf upload asserelzeki/UNIStainnet ./checkpoints/baseline_ki67/last.ckpt baseline_ki67_150epoch.ckpt


PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir /home/ahmed_ayman/data/Destained_MIST \
    --ckpt_dir "./checkpoints/ki67_deepliif_run4" \
    --stains Ki67 \
    --deepliif_weights_path deepliif-weights/DeepLIIF_Latest_Model/latest_net_G1.pth \
    --case_a_prob 1.0 \
    --case_a_warmup_epochs 40 \
    --case_a_anneal_epochs 80 \
    --case_a_end_prob 0 \
    --use_alignment \
    --batch_size 8 \
    --wandb_name "ki67_deepliif_run4" \
    --max_epochs 150 

    # --resume_from "./checkpoints/ki67_deepliif_run4/last.ckpt"
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/ki67_deepliif_run4/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains Ki67 \
  --output_dir "./eval_output/ki67_deepliif_run4_aligned" \
  --enable_stn_alignment 

PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir /home/ahmed_ayman/data/Destained_MIST \
    --ckpt_dir "./checkpoints/ki67_deepliif_run5" \
    --stains Ki67 \
    --deepliif_weights_path deepliif-weights/DeepLIIF_Latest_Model/latest_net_G1.pth \
    --case_a_prob 1.0 \
    --case_a_warmup_epochs 60 \
    --case_a_anneal_epochs 60 \
    --case_a_end_prob 0.05 \
    --use_alignment \
    --batch_size 8 \
    --wandb_name "ki67_deepliif_run5" \
    --max_epochs 150 
    # --resume_from "./checkpoints/ki67_deepliif_run5/last.ckpt"

PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/ki67_deepliif_run5/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains Ki67 \
  --output_dir "./eval_output/ki67_deepliif_run5_aligned" \
  --enable_stn_alignment 


PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir MIST \
    --ckpt_dir "./checkpoints/ki67_deepliif_run6_sharpness" \
    --stains Ki67 \
    --deepliif_weights_path deepliif-weights/DeepLIIF_Latest_Model/latest_net_G1.pth \
    --case_a_prob 1.0 \
    --case_a_warmup_epochs 40 \
    --case_a_anneal_epochs 80 \
    --case_a_end_prob 0 \
    --use_alignment \
    --batch_size 8 \
    --wandb_name "ki67_deepliif_run6_sharpness" \
    --max_epochs 150 

    # --resume_from "./checkpoints/ki67_deepliif_run6_sharpness"


PYTHONPATH=. python scripts/eval/eval_mist.py     --checkpoint "./checkpoints/ki67_deepliif_run3/last.ckpt"     --data_dir "/home/ahmed_ayman/data/Destained_MIST"     --stains Ki67     --output_dir "./eval_output/ki67_deepliif_run3"     --enable_stn_alignment     --ki67_eval_method deepliif     --seg_thresh 130     --marker_thresh default     --min_nuclei 100
PYTHONPATH=. python scripts/eval/eval_mist.py   --checkpoint "./checkpoints/baseline_ki67/last.ckpt"   --data_dir "/home/ahmed_ayman/data/Destained_MIST"   --stains Ki67   --output_dir "./eval_output/baseline_ki67_only"   --ki67_eval_method deepliif   --seg_thresh 130   --marker_thresh default   --min_nuclei 100
PYTHONPATH=. python scripts/eval/eval_mist.py   --checkpoint "./checkpoints/ki67_deepliif_run4/last.ckpt"   --data_dir "/home/ahmed_ayman/data/Destained_MIST"   --stains Ki67   --output_dir "./eval_output/ki67_deepliif_run4"   --enable_stn_alignment   --ki67_eval_method deepliif   --seg_thresh 130   --marker_thresh default   --min_nuclei 100
PYTHONPATH=. python scripts/eval/eval_mist.py   --checkpoint "./checkpoints/ki67_deepliif_run5/last.ckpt"   --data_dir "/home/ahmed_ayman/data/Destained_MIST"   --stains Ki67   --output_dir "./eval_output/ki67_deepliif_run5"   --enable_stn_alignment   --ki67_eval_method deepliif   --seg_thresh 130   --marker_thresh default   --min_nuclei 100
PYTHONPATH=. python scripts/eval/eval_mist.py   --checkpoint "./checkpoints/ki67_deepliif_run6_sharpness/last.ckpt"   --data_dir "/home/ahmed_ayman/data/Destained_MIST"   --stains Ki67   --output_dir "./eval_output/ki67_deepliif_run6_sharpness"   --enable_stn_alignment   --ki67_eval_method deepliif   --seg_thresh 130   --marker_thresh default   --min_nuclei 100 


PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir /home/ahmed_ayman/data/Destained_MIST \
    --ckpt_dir "./checkpoints/ki67_deepliif_run6" \
    --stains Ki67 \
    --deepliif_weights_path deepliif-weights/DeepLIIF_Latest_Model/latest_net_G1.pth \
    --case_a_prob 1.0 \
    --case_a_warmup_epochs 60 \
    --case_a_anneal_epochs 40 \
    --case_a_end_prob 0.05 \
    --use_alignment \
    --batch_size 8 \
    --wandb_name "ki67_deepliif_run6" \
    --max_epochs 150 
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/ki67_deepliif_run6/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains Ki67 \
  --output_dir "./eval_output/ki67_deepliif_run6" \
  --enable_stn_alignment \
  --ki67_eval_method deepliif \
  --seg_thresh 130  \
  --marker_thresh default   \
  --min_nuclei 100 


# Line ~120-128 — change these three:
  adversarial_weight=0.0,      # → 0.5
  gram_style_weight=0.0,       # → 0.5
  dab_contrast_weight=0.0,     # → 0.2

  Everything else stays exactly as Run 6.

  Then run:

  PYTHONPATH=. python scripts/train/train_mist.py \
      --data_dir /home/ahmed_ayman/data/Destained_MIST \
      --ckpt_dir "./checkpoints/ki67_deepliif_run7" \
      --stains Ki67 \
      --deepliif_weights_path deepliif-weights/DeepLIIF_Latest_Model/latest_net_G1.pth \
      --case_a_prob 1.0 \
      --case_a_warmup_epochs 60 \
      --case_a_anneal_epochs 40 \
      --case_a_end_prob 0.05 \
      --use_alignment \
      --batch_size 8 \
      --wandb_name "ki67_deepliif_run7" \
      --max_epochs 150
PYTHONPATH=. python scripts/eval/eval_mist.py \
  --checkpoint "./checkpoints/ki67_deepliif_run7/last.ckpt" \
  --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
  --stains Ki67 \
  --output_dir "./eval_output/ki67_deepliif_run7" \
  --enable_stn_alignment \
  --ki67_eval_method deepliif \
  --seg_thresh 130  \
  --marker_thresh default   \
  --min_nuclei 100

  ============================================================
  Ki67 CLINICAL EVALUATION  (DeepLIIF Gold Standard)
============================================================
  Images evaluated      : 891
  Real  LI (mean ± std) : 28.34% ± 23.19%
  Fake  LI (mean ± std) : 25.71% ± 22.69%
------------------------------------------------------------
  MAE (LI %)            : 10.10 ± 10.58
  Pearson r             : 0.8035  (p=1.78e-202)
  Tier Concordance      : 67.6%
  Weighted Kappa        : 0.5994
============================================================


  Ki67: FID=53.6 | KID=24.4 | LPIPS=0.509 | SSIM=0.281 | Pearson-r=0.881 | HE-H-SSIM=0.150 | HE-NMI=0.028
         Ki67: MAE=10.10% | r=0.804 | Concordance=67.6% | Kappa=0.599

======================================================================
MACRO-AVERAGED RESULTS
======================================================================
Metric                   Ki67    Macro
--------------------------------------
fid_inception          53.605   53.605
fid_uni               395.805  395.805
kid_mean_x1000         24.395   24.395
lpips_mean              0.509    0.509
lpips_128_mean          0.330    0.330
ssim_mean               0.281    0.281
psnr_mean              14.460   14.460
dab_mae_overall         0.140    0.140
dab_pearson_r           0.881    0.881
dab_kl                  0.219    0.219
dab_jsd                 0.049    0.049
miod_diff              -0.006   -0.006
miod_abs_diff           0.024    0.024
he_h_ssim_mean          0.150    0.150
he_nmi_mean             0.028    0.028
ki67_li_mae            10.096   10.096
ki67_li_pearson_r       0.804    0.804
ki67_tier_concordance    0.676    0.676
ki67_tier_kappa         0.599    0.599

Results saved to eval_output/ki67_deepliif_run7/results.json

PYTHONPATH=. python scripts/train/train_mist.py       --data_dir /home/ahmed_ayman/data/Destained_MIST       --ckpt_dir "./checkpoints/ki67_deepliif_run8"       --stains Ki67       --deepliif_weights_path deepliif-weights/DeepLIIF_Latest_Model/latest_net_G1.pth       --case_a_prob 1.0       --case_a_warmup_epochs 60       --case_a_anneal_epochs 40       --case_a_end_prob 0.05       --use_alignment       --batch_size 8       --wandb_name "ki67_deepliif_run8"       --max_epochs 150       --he_rgb_dropout 0.1
PYTHONPATH=. python scripts/eval/eval_mist.py   --checkpoint "./checkpoints/ki67_deepliif_run8/last.ckpt"   --data_dir "/home/ahmed_ayman/data/Destained_MIST"   --stains Ki67   --output_dir "./eval_output/ki67_deepliif_run8"   --enable_stn_alignment   --ki67_eval_method deepliif   --seg_thresh 80   --marker_thresh 80   --min_nuclei 80  --uni_fid_pooling all



# New Version V5
PYTHONPATH=. python scripts/train/train_mist.py \
      --data_dir /home/ahmed_ayman/data/Destained_MIST \
      --ckpt_dir "./checkpoints/ki67_pv1" \
      --stains Ki67 \
      --deepliif_weights_path deepliif-weights/DeepLIIF_Latest_Model/latest_net_G1.pth \
      --case_a_prob 1.0 \
      --case_a_warmup_epochs 60 \
      --case_a_anneal_epochs 40 \
      --case_a_end_prob 0.05 \
      --use_alignment \
      --batch_size 8 \
      --wandb_name "V5_ki67_pv1" \
      --max_epochs 150 \
      --he_rgb_dropout 0.1
PYTHONPATH=. python scripts/eval/eval_mist.py \
      --checkpoint "./checkpoints/ki67_pv1/last.ckpt" \
      --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
      --stains Ki67 \
      --output_dir "./eval_output/ki67_pv1" \
      --ki67_eval_method deepliif \
      --seg_thresh 80 \
      --marker_thresh 80 \
      --min_nuclei 80 \
      --uni_fid_pooling all


PYTHONPATH=. python scripts/train/train_mist.py \
      --data_dir /home/ahmed_ayman/data/Destained_MIST \
      --ckpt_dir "./checkpoints/ki67_pv2" \
      --stains Ki67 \
      --deepliif_weights_path deepliif-weights/DeepLIIF_Latest_Model/latest_net_G1.pth \
      --case_a_prob 1.0 \
      --case_a_warmup_epochs 60 \
      --case_a_anneal_epochs 40 \
      --case_a_end_prob 0.05 \
      --use_alignment \
      --batch_size 8 \
      --wandb_name "ki67_pv2" \
      --max_epochs 150 \
      --he_rgb_dropout 0.1
PYTHONPATH=. python scripts/eval/eval_mist.py \
      --checkpoint "./checkpoints/ki67_pv2/last.ckpt" \
      --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
      --stains Ki67 \
      --output_dir "./eval_output/ki67_pv2" \
      --ki67_eval_method deepliif \
      --seg_thresh 80 \
      --marker_thresh 80 \
      --min_nuclei 80 \
      --uni_fid_pooling all


PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir /home/ahmed_ayman/data/Destained_MIST \
    --ckpt_dir "./checkpoints/ki67_pv3" \
    --stains Ki67 \
    --deepliif_weights_path deepliif-weights/DeepLIIF_Latest_Model/latest_net_G1.pth \
    --case_a_prob 1.0 \
    --case_a_warmup_epochs 60 \
    --case_a_anneal_epochs 40 \
    --case_a_end_prob 0.05 \
    --use_alignment \
    --batch_size 8 \
    --wandb_name "ki67_pv3" \
    --max_epochs 150 \
    --he_rgb_dropout 0.1
PYTHONPATH=. python scripts/eval/eval_mist.py \
    --checkpoint "./checkpoints/ki67_pv3/last.ckpt" \
    --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
    --stains Ki67 \
    --output_dir "./eval_output/ki67_pv3" \
    --ki67_eval_method deepliif \
    --seg_thresh 80 \
    --marker_thresh 80 \
    --min_nuclei 80 \
    --uni_fid_pooling all

PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir /home/ahmed_ayman/data/Destained_MIST \
    --ckpt_dir "./checkpoints/ki67_pv4" \
    --stains Ki67 \
    --deepliif_weights_path deepliif-weights/DeepLIIF_Latest_Model/latest_net_G1.pth \
    --case_a_prob 1.0 \
    --case_a_warmup_epochs 60 \
    --case_a_anneal_epochs 40 \
    --case_a_end_prob 0.05 \
    --use_alignment \
    --batch_size 8 \
    --wandb_name "ki67_pv4" \
    --max_epochs 150 \
    --he_rgb_dropout 0.1
PYTHONPATH=. python scripts/eval/eval_mist.py \
    --checkpoint "./checkpoints/ki67_pv4/last.ckpt" \
    --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
    --stains Ki67 \
    --output_dir "./eval_output/ki67_pv4" \
    --ki67_eval_method deepliif \
    --seg_thresh 80 \
    --marker_thresh 80 \
    --min_nuclei 80 \
    --uni_fid_pooling all


  Run 9 — Launch Now

  PYTHONPATH=. python scripts/train/train_mist.py \
      --data_dir /home/ahmed_ayman/data/Destained_MIST \
      --ckpt_dir "./checkpoints/ki67_run9" --stains Ki67 \
      --deepliif_weights_path deepliif-weights/DeepLIIF_Latest_Model/latest_net_G1.pth \
      --case_a_prob 1.0 --case_a_warmup_epochs 60 --case_a_anneal_epochs 40 --case_a_end_prob 0.05 \
      --use_alignment --batch_size 8 --wandb_name "ki67_run9" --max_epochs 150 --he_rgb_dropout 0.1 \
      --resume_from "./checkpoints/ki67_run9/last.ckpt"
  PYTHONPATH=. python scripts/eval/eval_mist.py \
      --checkpoint "./checkpoints/ki67_run9/last.ckpt" \
      --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
      --stains Ki67 \
      --output_dir "./eval_output/ki67_run9" \
      --enable_stn_alignment \
      --ki67_eval_method deepliif \
      --seg_thresh 80 \
      --marker_thresh 80 \
      --min_nuclei 80 \
      --uni_fid_pooling all
  
  
  Config: Run 8 + MLPA(0.5) + GP(1.0) − L1_lowres − DAB_intensity. class_dim=64, gram=0, patchnce=0.

  Run 10 — Launch After Run 9

  PYTHONPATH=. python scripts/train/train_mist.py \
      --data_dir /home/ahmed_ayman/data/Destained_MIST \
      --ckpt_dir "./checkpoints/ki67_run10" --stains Ki67 \
      --deepliif_weights_path deepliif-weights/DeepLIIF_Latest_Model/latest_net_G1.pth \
      --case_a_prob 1.0 --case_a_warmup_epochs 60 --case_a_anneal_epochs 40 --case_a_end_prob 0.05 \
      --use_alignment --batch_size 8 --wandb_name "ki67_run10" --max_epochs 150 --he_rgb_dropout 0.1 \
      --resume_from "./checkpoints/ki67_run10/last.ckpt"
  PYTHONPATH=. python scripts/eval/eval_mist.py \
      --checkpoint "./checkpoints/ki67_run10/last.ckpt" \
      --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
      --stains Ki67 \
      --output_dir "./eval_output/ki67_run10" \
      --enable_stn_alignment \
      --ki67_eval_method deepliif \
      --seg_thresh 80 \
      --marker_thresh 80 \
      --min_nuclei 80 \
      --uni_fid_pooling all

  Config: Same as Run 9 + patchnce_weight=0.1. Already set in the code. Same command as Run 9 but different --ckpt_dir and --wandb_name.

● # Run 9
  PYTHONPATH=. python scripts/eval/eval_mist.py \
      --checkpoint "./checkpoints/ki67_run9/last.ckpt" \
      --data_dir "/home/ahmed_ayman/data/Destained_MIST" --stains Ki67 \
      --output_dir "./eval_output/ki67_run9" \
      --ki67_eval_method deepliif \
      --seg_thresh 80 --marker_thresh 80 --min_nuclei 80

  # Run 10
  PYTHONPATH=. python scripts/eval/eval_mist.py \
      --checkpoint "./checkpoints/ki67_run10/last.ckpt" \
      --data_dir "/home/ahmed_ayman/data/Destained_MIST" --stains Ki67 \
      --output_dir "./eval_output/ki67_run10" \
      --ki67_eval_method deepliif \
      --seg_thresh 80 --marker_thresh 80 --min_nuclei 80

###########################################################################################################

Run 11 = Run 10 Config + Two Fixes

  Both fixes are already in the code (I made them after Run 10 launched):

  1. Misalignment-aware pyramid weights: 512 weight 1.0→0.5, 256 weight 0.25→0.75. Less penalty for correctly-placed but shifted nuclei.
  2. Tiny L1 anchor: l1_lowres_weight=0.05. A whisper of color guidance — "stay roughly this brown" — without the hallucination pressure of the old 0.5 weight.

  Launch with the same command as Run 10 but different --ckpt_dir and --wandb_name:

  PYTHONPATH=. python scripts/train/train_mist.py \
      --data_dir /home/ahmed_ayman/data/Destained_MIST \
      --deepliif_weights_path deepliif-weights/DeepLIIF_Latest_Model/latest_net_G1.pth \
      --case_a_prob 1.0 --case_a_warmup_epochs 60 --case_a_anneal_epochs 40 --case_a_end_prob 0.05 \
      --use_alignment --batch_size 8 --wandb_name "ki67_run11" --max_epochs 150 --he_rgb_dropout 0.1 \
      --ckpt_dir "./checkpoints/ki67_run11" --stains Ki67         

#############################################################################################################
● Done. Run 11 now has:

  ┌──────────────┬───────────────┬───────────────────────────────────────────────┐
  │    Change    │     Lines     │                     What                      │
  ├──────────────┼───────────────┼───────────────────────────────────────────────┤
  │ SE-Net ×5    │ ~80           │ Channel attention after each decoder block    │
  ├──────────────┼───────────────┼───────────────────────────────────────────────┤
  │ CoordAttn ×1 │ ~40           │ Factorised spatial attention at bottleneck    │
  ├──────────────┼───────────────┼───────────────────────────────────────────────┤
  │ GP weights   │ fixed earlier │ 256×256 peak (0.75), 512×512 reduced (0.5)    │
  ├──────────────┼───────────────┼───────────────────────────────────────────────┤
  │ L1 anchor    │ 1 line        │ l1_lowres_weight=0.05 — whisper of color      │
  ├──────────────┼───────────────┼───────────────────────────────────────────────┤
  │ PatchNCE     │ 1 line        │ patchnce_weight=0.1 — feature-space structure │
  └──────────────┴───────────────┴───────────────────────────────────────────────┘

  Total params added: ~50K (<0.2% increase). No new hyperparameters.

  PYTHONPATH=. python scripts/train/train_mist.py \
      --data_dir /home/ahmed_ayman/data/Destained_MIST \
      --ckpt_dir "./checkpoints/ki67_run11" --stains Ki67 \
      --deepliif_weights_path deepliif-weights/DeepLIIF_Latest_Model/latest_net_G1.pth \
      --case_a_prob 1.0 --case_a_warmup_epochs 60 --case_a_anneal_epochs 40 --case_a_end_prob 0.05 \
      --use_alignment --batch_size 8 --wandb_name "ki67_run11" --max_epochs 150 --he_rgb_dropout 0.1 \
      --resume_from "./checkpoints/ki67_run11/last.ckpt"

  PYTHONPATH=. python scripts/eval/eval_mist.py \
      --checkpoint "./checkpoints/ki67_run11/last.ckpt" \
      --data_dir "/home/ahmed_ayman/data/Destained_MIST" \
      --stains Ki67 \
      --output_dir "./eval_output/ki67_run11" \
      --enable_stn_alignment \
      --ki67_eval_method deepliif \
      --seg_thresh 80 \
      --marker_thresh 80 \
      --min_nuclei 80 
################################################################################################################
  Run 11 — Weights Only

  # Same as Run 10 + two fixes:
  gp_weights = [0.1, 0.15, 0.25, 0.5, 0.75, 0.5]  # misalignment-aware
  l1_lowres_weight = 0.05                            # tiny color anchor
  patchnce_weight = 0.1                              # feature-space structure
  use_se_attention = False                           # no architecture changes


  Run 12 — Architecture

  Change one line in train_mist.py:
  use_se_attention=True,   # Run 12: on

  PYTHONPATH=. python scripts/train/train_mist.py \
      --data_dir /home/ahmed_ayman/data/Destained_MIST \
      --ckpt_dir "./checkpoints/ki67_run12" --stains Ki67 \
      --deepliif_weights_path deepliif-weights/DeepLIIF_Latest_Model/latest_net_G1.pth \
      --case_a_prob 1.0 --case_a_warmup_epochs 60 --case_a_anneal_epochs 40 --case_a_end_prob 0.05 \
      --use_alignment --batch_size 8 --wandb_name "ki67_run12" --max_epochs 120 --he_rgb_dropout 0.1

  Then same command with different --ckpt_dir and --wandb_name.
####################################################################################################################

  PYTHONPATH=. python scripts/train/train_mist.py \
      --data_dir /home/ahmed_ayman/data/Destained_MIST \
      --ckpt_dir "./checkpoints/ki67_run12" --stains Ki67 \
      --deepliif_weights_path deepliif-weights/DeepLIIF_Latest_Model/latest_net_G1.pth \
      --case_a_prob 1.0 --case_a_warmup_epochs 60 --case_a_anneal_epochs 40 --case_a_end_prob 0.05 \
      --use_alignment --batch_size 8 --wandb_name "ki67_run12" --max_epochs 150 --he_rgb_dropout 0.1 \
        --resume_from "./checkpoints/ki67_run12/last.ckpt"


# hf upload asserelzeki/run12 "./checkpoints/ki67_run12/last.ckpt" run12_epoch81.ckpt
