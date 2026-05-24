#!/bin/bash
# =============================================================================
# UNIStainNet Destaining Training — Universal Launch Script
#
# Usage:
#   ./run_training.sh [-s "STAIN1 STAIN2 ..."] [-z /path/to/zips]
#
# Examples:
#   ./run_training.sh                                 (Runs all 4 default stains)
#   ./run_training.sh -s "HER2"                       (Runs only HER2)
#   ./run_training.sh -s "HER2 ER" -z "/custom/path"  (Runs HER2 & ER from custom zip path)
# =============================================================================

set -e   # stop immediately on any error

# 1. Dynamically detect the repository root (assumes this script is in the repo root)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 2. Default Variables
ZIP_DIR="./Destained_Results"
STAINS="ER PR"
# STAINS="HER2 ER Ki67 PR"


# 3. Parse command-line arguments for custom stains or zip locations
while getopts z:s: flag
do
    case "${flag}" in
        z) ZIP_DIR=${OPTARG};;
        s) STAINS=${OPTARG};;
        *) echo "Usage: $0 [-s \"STAINS\"] [-z ZIP_DIR]"; exit 1;;
    esac
done
echo "Selected stains: $STAINS"

# 4. Set internal directories (strictly inside the repo folder)
DATA_DIR="./data/MIST"
CKPT_DIR="./checkpoints/destaining_v2_ER_PR_Only"

echo "========================================================"
echo "  Configuration Overview"
echo "========================================================"
echo "  Repo Directory:  $REPO_DIR"
echo "  Zip Directory:   $ZIP_DIR"
echo "  Data Directory:  $DATA_DIR"
echo "  Checkpoints:     $CKPT_DIR"
echo "  Selected Stains: $STAINS"
echo "========================================================"

echo ""
echo "========================================================"
echo "  Step 0: Install Required Datasets"
echo "========================================================"

mkdir -p "$ZIP_DIR"

for STAIN in $STAINS; do
    ZIP_NAME="${STAIN}-Destained.zip"
    LOCAL_ZIP="$ZIP_DIR/$ZIP_NAME"

    if [ -f "$LOCAL_ZIP" ]; then
        echo "  [SKIP] $ZIP_NAME already exists in $ZIP_DIR"
    else
        echo "  [DOWNLOADING] $ZIP_NAME from Hugging Face..."
        
        # Using huggingface-cli to download only the specific file
        # This works without login IF the dataset is Public.
        hf download asserelzeki/destained-histopathology-data \
            "$ZIP_NAME" \
            --repo-type dataset \
            --local-dir "$ZIP_DIR" \
            --local-dir-use-symlinks False
            
        echo "  [OK] Finished downloading $STAIN"
    fi
done

echo ""
echo "========================================================"
echo "  Step 1: Unzip and rename stain folders"
echo "========================================================"

mkdir -p "$DATA_DIR"

# Loop only through the user-selected stains
for STAIN in $STAINS; do
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

for STAIN in $STAINS; do
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

echo ""
echo "========================================================"
echo "  Step 3: Install dependencies"
echo "========================================================"

cd "$REPO_DIR"
pip install -r requirements.txt -q
pip install -e . -q
echo "  Dependencies installed."

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

# # Passes the dynamic list of stains to the python script
# python scripts/debug/validate_data.py \
#     --data_dir "$DATA_DIR" \
#     --stains $STAINS \
#     --batch_size 4 \
#     --n_batches 3
# echo "  Data loading check passed."

echo ""
echo "========================================================"
echo "  Step 6: Train (GPU required from this point)"
echo "========================================================"
# Adjust --batch_size based on GPU:
#   A10G  24GB  → 4–6
#   A100  40GB  → 8       ← recommended starting point
#   A100  80GB  → 12–16

mkdir -p "$CKPT_DIR"

# python scripts/train/train_mist.py \
#     --data_dir   "$DATA_DIR" \
#     --stains     $STAINS \
#     --batch_size 8 \
#     --max_epochs 100 \
#     --ckpt_dir   "$CKPT_DIR" \
#     --wandb_name destaining_v1_attention_batch8_${STAINS// /_}

PYTHONPATH=. python scripts/train/train_mist.py \
    --data_dir   "$DATA_DIR" \
    --stains     $STAINS \
    --batch_size 8 \
    --ckpt_dir "checkpoints/destaining_v2_attn_residual_only" \
    --edge_encoder v2 \
    --enable_attention_residual \
    --use_eosin_encoder \
    --no_spade_use_uni \
    --no_use_attention_for_spade  \
    --wandb_name destaining_v2_attention_batch4_ER_PR_Only_residual_Only

# For Resuming from a checkpoint, use the following command (uncomment and adjust the --ckpt_dir and --wandb_name as needed):
# PYTHONPATH=. python scripts/train/train_mist.py \
#     --data_dir   "$DATA_DIR" \
#     --stains     $STAINS \
#     --batch_size 8 \
#     --ckpt_dir "checkpoints/destaining_v2_attn_residual_only" \
#     --edge_encoder v2 \
#     --enable_attention_residual \
#     --use_eosin_encoder \
#     --no_spade_use_uni \
#     --no_use_attention_for_spade  \
#     --wandb_name destaining_v2_attention_batch4_ER_PR_Only_residual_Only \
#     --log_val_fid \
#     --resume_from "checkpoints/destaining_v2_attn_residual_only/last.ckpt"

echo "  Training process initiated."