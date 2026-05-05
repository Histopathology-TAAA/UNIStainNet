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
REPO_DIR="$STUDIO_ROOT/UNISTAINNET"
ZIP_DIR="$STUDIO_ROOT/Destained_Results"
DATA_DIR="$STUDIO_ROOT/data/MIST"

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

echo ""
echo "========================================================"
echo "  Step 3: Install dependencies"
echo "========================================================"

cd "$REPO_DIR"
pip install -r requirements.txt -q
pip install -e . -q
echo "  Dependencies installed."

echo ""
echo "========================================================"
echo "  Step 4: CPU sanity check (no GPU, no UNI download)"
echo "========================================================"

export PYTHONPATH="$REPO_DIR"
python scripts/debug/sanity_check_mist_dummy.py
echo "  Architecture sanity check passed."

echo ""
echo "========================================================"
echo "  Step 5: Real data loading check (CPU, no model)"
echo "========================================================"

python scripts/debug/validate_data.py \
    --data_dir "$DATA_DIR" \
    --stains HER2 ER Ki67 PR \
    --batch_size 4 \
    --n_batches 3
echo "  Data loading check passed."

echo ""
echo "========================================================"
echo "  Step 6: Train  (GPU required from this point)"
echo "========================================================"
# Adjust --batch_size based on GPU:
#   A10G  24GB  → 4–6
#   A100  40GB  → 8       ← recommended starting point
#   A100  80GB  → 12–16

python scripts/train/train_mist.py \
    --data_dir   "$DATA_DIR" \
    --stains     HER2 ER Ki67 PR \
    --batch_size 8 \
    --max_epochs 100 \
    --ckpt_dir   "$STUDIO_ROOT/checkpoints/destaining_v1" \
    --wandb_name destaining_v1_attention_batch8

# For copy and paste command running uncomment, copy, and use this code 

# python scripts/train/train_mist.py \
#     --data_dir   "$HOME/data/MIST" \
#     --stains     HER2 ER Ki67 PR \
#     --batch_size 8 \
#     --max_epochs 100 \
#     --ckpt_dir   "$HOME/checkpoints/destaining_v1" \
#     --wandb_name destaining_v1_attention_batch8


# change the wandb name if needed to log 