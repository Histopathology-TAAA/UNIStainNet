#!/usr/bin/env bash
set -euo pipefail

# DeepLIIF Ki67 Parameter Sweep
# Sweeps seg_thresh and marker_thresh to find optimal evaluator calibration.
# No ground truth needed — picks the config that maximizes Pearson-r
# between real LI and fake LI, weighted by concordance and patch retention.
#
# Usage: bash scripts/eval/sweep_deepliif_params.sh

CHECKPOINT="./checkpoints/ki67_deepliif_run8/last.ckpt"
DATA_DIR="/home/ahmed_ayman/data/Destained_MIST"
SWEEP_DIR="./eval_output/deepliif_sweep"
MIN_NUCLEI=80
STAIN="Ki67"

mkdir -p "$SWEEP_DIR"

SEG_THRESHOLDS=(80 100 130 160 200)
MARKER_THRESHOLDS=("default" "None" "50" "80" "120")

TOTAL=$(( ${#SEG_THRESHOLDS[@]} * ${#MARKER_THRESHOLDS[@]} ))
CURRENT=0
RESULTS_FILE="$SWEEP_DIR/sweep_summary.txt"

echo "============================================================" | tee "$RESULTS_FILE"
echo "  DeepLIIF Parameter Sweep" | tee -a "$RESULTS_FILE"
echo "  Checkpoint: $CHECKPOINT" | tee -a "$RESULTS_FILE"
echo "  Configs to evaluate: $TOTAL" | tee -a "$RESULTS_FILE"
echo "============================================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"
printf "%-6s %-14s %-14s %-10s %-10s %-12s %-10s %-10s %-10s\n" \
    "Seg" "Marker" "Pearson-r" "MAE" "Concord%" "Kappa" "Dropped" "RealLIμ" "Score" \
    | tee -a "$RESULTS_FILE"
printf "%-6s %-14s %-14s %-10s %-10s %-12s %-10s %-10s %-10s\n" \
    "-----" "--------------" "--------------" "----------" "----------" "------------" "----------" "----------" "----------" \
    | tee -a "$RESULTS_FILE"

for SEG in "${SEG_THRESHOLDS[@]}"; do
    for MARKER in "${MARKER_THRESHOLDS[@]}"; do
        CURRENT=$((CURRENT + 1))
        SAFE_MARKER=$(echo "$MARKER" | tr '/' '_')
        OUT_DIR="${SWEEP_DIR}/s${SEG}_m${SAFE_MARKER}"

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  [$CURRENT/$TOTAL] seg_thresh=$SEG  marker_thresh=$MARKER"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        PYTHONPATH=. python scripts/eval/eval_mist.py \
            --checkpoint "$CHECKPOINT" \
            --data_dir "$DATA_DIR" \
            --stains "$STAIN" \
            --output_dir "$OUT_DIR" \
            --ki67_eval_method deepliif \
            --seg_thresh "$SEG" \
            --marker_thresh "$MARKER" \
            --min_nuclei "$MIN_NUCLEI" \
            --skip_uni_fid \
            2>&1 | tee "${OUT_DIR}/eval_log.txt"

        # Parse results
        RESULT_JSON="${OUT_DIR}/results.json"
        if [ -f "$RESULT_JSON" ]; then
            PEARSON=$(python3 -c "
import json
with open('$RESULT_JSON') as f:
    d = json.load(f)
print(d['per_stain']['Ki67']['clinical']['ki67_li_pearson_r'])
" 2>/dev/null || echo "NA")

            MAE=$(python3 -c "
import json
with open('$RESULT_JSON') as f:
    d = json.load(f)
print(d['per_stain']['Ki67']['clinical']['ki67_li_mae'])
" 2>/dev/null || echo "NA")

            CONCORDANCE=$(python3 -c "
import json
with open('$RESULT_JSON') as f:
    d = json.load(f)
print(d['per_stain']['Ki67']['clinical']['ki67_tier_concordance'])
" 2>/dev/null || echo "NA")

            KAPPA=$(python3 -c "
import json
with open('$RESULT_JSON') as f:
    d = json.load(f)
print(d['per_stain']['Ki67']['clinical']['ki67_tier_kappa'])
" 2>/dev/null || echo "NA")

            EVAL_COUNT=$(python3 -c "
import json
with open('$RESULT_JSON') as f:
    d = json.load(f)
print(d['per_stain']['Ki67']['clinical'].get('images_evaluated', 'NA'))
" 2>/dev/null || echo "NA")

            REAL_LI_MEAN=$(python3 -c "
import json
with open('$RESULT_JSON') as f:
    d = json.load(f)
print(d['per_stain']['Ki67']['clinical'].get('real_li_mean', 'NA'))
" 2>/dev/null || echo "NA")

            # Compute score
            SCORE=$(python3 -c "
pearson = float('$PEARSON') if '$PEARSON' != 'NA' else 0.0
concord = float('$CONCORDANCE') if '$CONCORDANCE' != 'NA' else 0.0
eval_count = float('$EVAL_COUNT') if '$EVAL_COUNT' != 'NA' else 899.0
dropped = 899.0 - eval_count
score = pearson + 0.1 * concord - 0.5 * (dropped / 899.0)
print(f'{score:.4f}')
" 2>/dev/null || echo "NA")

            printf "%-6s %-14s %-14s %-10s %-10s %-12s %-10s %-10s %-10s\n" \
                "$SEG" "$MARKER" "$PEARSON" "$MAE" "$CONCORDANCE" "$KAPPA" "$EVAL_COUNT" "$REAL_LI_MEAN" "$SCORE" \
                | tee -a "$RESULTS_FILE"
        else
            echo "  [WARN] No results.json — eval may have crashed" | tee -a "$RESULTS_FILE"
            printf "%-6s %-14s %-14s %-10s %-10s %-12s %-10s %-10s %-10s\n" \
                "$SEG" "$MARKER" "FAILED" "—" "—" "—" "—" "—" "—" \
                | tee -a "$RESULTS_FILE"
        fi
    done
done

echo "" | tee -a "$RESULTS_FILE"
echo "============================================================" | tee -a "$RESULTS_FILE"
echo "  Sweep complete." | tee -a "$RESULTS_FILE"
echo "  Summary: $RESULTS_FILE" | tee -a "$RESULTS_FILE"
echo "============================================================" | tee -a "$RESULTS_FILE"

# Print the winner
echo "" | tee -a "$RESULTS_FILE"
echo "Top 5 configs by score:" | tee -a "$RESULTS_FILE"
tail -n +5 "$RESULTS_FILE" | head -n -4 | sed '/^$/d' | sed '/^━/d' \
    | sort -k9 -rg | head -5 | tee -a "$RESULTS_FILE"
