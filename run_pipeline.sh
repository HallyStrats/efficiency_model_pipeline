#!/bin/bash
# run_pipeline.sh — Orchestrator for the src_v6 pipeline
#
# Usage: bash run_pipeline.sh [step]
#   No args: run all steps 01-08
#   step:    run from that step onward (e.g., "03" runs 03-08)
#
# Full pipeline reproduces the best GPS-only energy consumption model:
#   01  -> Ingest raw CSVs
#   02  -> Clean and validate
#   03  -> Feature engineering (GPS + sensor + target)
#   04  -> Segment into micro-trips
#   05  -> Train/val/test split by rider
#   06  -> Train baseline GPS and full-feature models
#   06b -> Build OSM road type cache from OpenStreetMap
#   06c -> Train best GPS model (LSTM + tuned XGBoost, Wh/km target, P90 clip)
#   07  -> Train full-feature models
#   08  -> Evaluate all models and generate plots/metrics

set -e
cd "$(dirname "$0")"

START="${1:-01}"

run_step() {
    local step="$1"
    local script="$2"
    echo ""
    echo "================================================================"
    echo "  Step $step: $script"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
    local t0=$(date +%s)
    python3 "$script"
    local t1=$(date +%s)
    echo ""
    echo "  Completed in $((t1 - t0)) seconds"
    echo "================================================================"
}

STEPS=(
    "01:01_ingest.py"
    "02:02_clean.py"
    "03:03_features.py"
    "04:04_segment.py"
    "05:05_split.py"
    "06:06_train_gps.py"
    "06b:06b_osm_cache.py"
    "06c:06c_train_best.py"
    "07:07_train_full.py"
    "08:08_evaluate.py"
)

echo "========================================"
echo "  src_v6 Pipeline"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

TOTAL_START=$(date +%s)

for entry in "${STEPS[@]}"; do
    step="${entry%%:*}"
    script="${entry#*:}"
    if [[ "$step" >= "$START" ]]; then
        run_step "$step" "$script"
    fi
done

TOTAL_END=$(date +%s)
echo ""
echo "========================================"
echo "  Pipeline complete"
echo "  Total time: $((TOTAL_END - TOTAL_START)) seconds"
echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
