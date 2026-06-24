#!/usr/bin/env bash
# Download pretrained models for UNIStainNet PGVMS integration.
#
# Required:
#   MIST_Ki67_net_seg.pth  —  Frozen tumour-segmentation UNet for CTPC/PCLS loss
#
# Source: https://drive.google.com/drive/folders/1ekuPcvVLlX0D0IQ-OIHdh5L3zwnU-s4l?usp=sharing
#
# Usage:
#   bash scripts/download_pretrained.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRETRAIN_DIR="$SCRIPT_DIR/../pretrain"
mkdir -p "$PRETRAIN_DIR"

echo "========================================"
echo "  Downloading PGVMS pretrained models"
echo "  Target: $PRETRAIN_DIR"
echo "========================================"

if ! command -v gdown &> /dev/null; then
    echo "[INSTALL] gdown not found. Installing..."
    pip install gdown
fi

echo ""
echo "[DOWNLOAD] Fetching from Google Drive..."
gdown --folder "https://drive.google.com/drive/folders/1ekuPcvVLlX0D0IQ-OIHdh5L3zwnU-s4l" \
    -O "$PRETRAIN_DIR/"

echo ""
echo "========================================"
echo "  Downloaded files:"
echo "========================================"
ls -lh "$PRETRAIN_DIR/"*.pth 2>/dev/null || ls -lhR "$PRETRAIN_DIR/"

echo ""
echo "========================================"
echo "  Flattening nested folders..."
echo "========================================"
# gdown --folder creates nested dirs: pretrain/pretrain/*.pth
# Move everything into pretrain/ directly
find "$PRETRAIN_DIR" -mindepth 2 -name "*.pth" -exec mv {} "$PRETRAIN_DIR/" \; 2>/dev/null || true
# Clean up empty subdirectories
find "$PRETRAIN_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null || true

echo ""
echo "========================================"
echo "  Downloaded files:"
echo "========================================"
ls -lh "$PRETRAIN_DIR/"*.pth 2>/dev/null || echo "  (no .pth files found)"

echo ""
echo "========================================"
echo "  Verification"
echo "========================================"
REQUIRED="$PRETRAIN_DIR/MIST_KI67_net_seg.pth"
if [ -f "$REQUIRED" ]; then
    echo "  [OK] $REQUIRED"
else
    echo "  [MISSING] $REQUIRED — check ls output above"
fi

# Rename to match what the trainer expects (Ki67 → KI67 casing)
if [ -f "$PRETRAIN_DIR/MIST_KI67_net_seg.pth" ] && [ ! -f "$PRETRAIN_DIR/MIST_Ki67_net_seg.pth" ]; then
    echo "  Note: trainer expects MIST_Ki67_net_seg.pth — file is MIST_KI67_net_seg.pth"
    echo "  The trainer was updated to use MIST_KI67_net_seg.pth (uppercase KI67)"
fi

echo ""
echo "  Ready for PV2 training:"
echo "    bash scripts/train/train_mist.py ... --wandb_name ki67_pv2 ..."
