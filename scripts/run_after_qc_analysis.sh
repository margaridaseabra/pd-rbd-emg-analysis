#!/usr/bin/env bash
set -euo pipefail

cd "/Users/margaridaseabra/Documents/GitHub/pd-rbd-emg-analysis-clean"

EVENTS="/Volumes/T7/Margarida/RBD-KatiasData/processed_all_weeks/qc_ready/EMG_episodes_NREMbaseline_all_weeks_qc_ready.csv"

OLD_EEG="/Volumes/T7/Margarida/RBD-KatiasData/processed_all_weeks/manifests/WT_PD_week2_week21_EEGonly_finalWTref_application_paths_fixed.csv"

NEW_EEG="data/manifests/new_weeks_somnotate_EEGonly_local_all_compatible.csv"

OUT_ROOT="/Volumes/T7/Margarida/RBD-KatiasData/processed_all_weeks/presentation_ready_outputs/$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUT_ROOT"

echo "Writing outputs to:"
echo "$OUT_ROOT"

echo ""
echo "1. REM-normalized automatic metrics"

python Code/analyze_all_weeks_rem_normalized_metrics.py \
  --events "$EVENTS" \
  --eegonly-manifests "$OLD_EEG" "$NEW_EEG" \
  --out-dir "$OUT_ROOT/1_REM_normalized_automatic_metrics"

echo ""
echo "2. Baseline-change analysis"

python Code/analyze_REM_normalized_baseline_change.py \
  --metrics "$OUT_ROOT/1_REM_normalized_automatic_metrics/mouse_week_REM_normalized_EMG_metrics.csv" \
  --out-dir "$OUT_ROOT/2_baseline_change"

echo ""
echo "3. QC-confirmed metrics"

python Code/analyze_qc_confirmed_metrics.py \
  --events "$EVENTS" \
  --rem-opportunity "$OUT_ROOT/1_REM_normalized_automatic_metrics/REM_opportunity_by_mouse_week.csv" \
  --out-dir "$OUT_ROOT/3_QC_confirmed_metrics"

echo ""
echo "4. Morphology summary, if morphology table exists"

if [ -f "/Volumes/T7/Margarida/RBD-KatiasData/processed_all_weeks/morphology/all_weeks_emg_morphology_features_with_clusters.csv" ]; then
  python Code/merge_morphology_into_all_weeks.py \
    --events "$EVENTS" \
    --morphology "/Volumes/T7/Margarida/RBD-KatiasData/processed_all_weeks/morphology/all_weeks_emg_morphology_features_with_clusters.csv" \
    --rem-opportunity "$OUT_ROOT/1_REM_normalized_automatic_metrics/REM_opportunity_by_mouse_week.csv" \
    --out-dir "$OUT_ROOT/4_morphology"
else
  echo "Morphology table not found yet. Skipping morphology."
fi

echo ""
echo "5. Spindle summary, if spindle metrics exist"

if [ -f "/Volumes/T7/Margarida/RBD-KatiasData/processed_all_weeks/spindles/all_weeks_spindle_mouse_week_metrics.csv" ]; then
  mkdir -p "$OUT_ROOT/5_spindles"
  cp "/Volumes/T7/Margarida/RBD-KatiasData/processed_all_weeks/spindles/all_weeks_spindle_mouse_week_metrics.csv" "$OUT_ROOT/5_spindles/"
else
  echo "Spindle metrics not found yet. Skipping spindles."
fi

echo ""
echo "Done."
echo "Open:"
echo "$OUT_ROOT"

open "$OUT_ROOT"
