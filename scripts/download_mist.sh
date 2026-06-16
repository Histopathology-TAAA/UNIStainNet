#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Target folder
TARGET_DIR="MIST"
mkdir -p "$TARGET_DIR"

# Files to download
FILES=("ER.zip" "HER2.zip" "Ki67.zip" "PR.zip")
REPO_ID="asserelzeki/MIST"

echo "============================================================"
echo "Downloading MIST dataset from Hugging Face: $REPO_ID"
echo "============================================================"

# Check if it's a dataset or model repository
echo "Detecting repository type..."
REPO_TYPE="dataset"
if hf download "$REPO_ID" "${FILES[0]}" --repo-type dataset --local-dir "$TARGET_DIR" > /dev/null 2>&1; then
    echo "[INFO] Detected Hugging Face Dataset repository."
else
    echo "[INFO] Not a dataset repo. Trying as a Model repository..."
    REPO_TYPE="model"
fi

for FILE in "${FILES[@]}"; do
    echo "------------------------------------------------------------"
    echo "Downloading $FILE..."
    
    # Download the file locally to the target directory
    hf download "$REPO_ID" "$FILE" \
        --repo-type "$REPO_TYPE" \
        --local-dir "$TARGET_DIR"

    echo "Extracting $FILE..."
    # Use python's built-in zip extractor to ensure compatibility
    python3 -m zipfile -e "$TARGET_DIR/$FILE" "$TARGET_DIR"

    echo "Deleting $FILE..."
    rm "$TARGET_DIR/$FILE"
done

echo "============================================================"
echo "SUCCESS: All files downloaded, unzipped, and cleaned up!"
echo "Dataset folder: $(pwd)/$TARGET_DIR"
echo "============================================================"
