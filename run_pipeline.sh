#!/bin/bash
# run_pipeline.sh — Orchestrator for the src_v6 pipeline
#
# Usage:
#   bash run_pipeline.sh              Resume — skips steps that are already done
#   bash run_pipeline.sh 06c          Force-rerun from step 06c onward
#   bash run_pipeline.sh --clean      Wipe all done markers (triggers full rerun)
#
# Resume logic:
#   A hidden marker file (.pipeline_done/step_XX) is written after each step
#   succeeds.  On restart, any step whose marker exists is skipped.  If a step
#   was interrupted mid-run, its marker was never written, so it reruns.
#
#   To force a single step to redo: rm .pipeline_done/step_06c && bash run_pipeline.sh
#
# Steps:
#   01  -> Ingest raw CSVs
#   02  -> Clean and validate
#   03  -> Feature engineering (GPS + sensor + target)
#   04  -> Segment into micro-trips
#   05  -> Train/val/test split by rider
#   06  -> Train baseline GPS models
#   06b -> Build OSM road type cache from OpenStreetMap
#   06c -> Train best GPS model (LSTM 3×seeds + tuned XGBoost, Wh/km target, P90 clip)
#   06d -> P90 clip sensitivity study (XGBoost at no_clip/P95/P90/P85)
#   07  -> Train full-feature models
#   08  -> Evaluate all models and generate plots/metrics

set -e
cd "$(dirname "$0")"

DONE_DIR=".pipeline_done"
mkdir -p "$DONE_DIR"

STEPS=(
    "01:01_ingest.py"
    "02:02_clean.py"
    "03:03_features.py"
    "04:04_segment.py"
    "05:05_split.py"
    "06:06_train_gps.py"
    "06b:06b_osm_cache.py"
    "06c:06c_train_best.py"
    "06d:06d_clip_sensitivity.py"
    "07:07_train_full.py"
    "08:08_evaluate.py"
)

# --clean: wipe all markers and exit
if [[ "$1" == "--clean" ]]; then
    rm -f "$DONE_DIR"/step_*
    echo "All done markers cleared — next run will execute the full pipeline."
    exit 0
fi

# Explicit start step: clear markers for that step and everything after it,
# then proceed normally (earlier steps stay marked as done and are skipped).
if [[ -n "$1" ]]; then
    START="$1"
    cleared=()
    for entry in "${STEPS[@]}"; do
        s="${entry%%:*}"
        if [[ "$s" > "$START" || "$s" == "$START" ]]; then
            rm -f "$DONE_DIR/step_$s"
            cleared+=("$s")
        fi
    done
    echo "Cleared done markers for steps: ${cleared[*]}"
else
    START="01"
fi

run_step() {
    local step="$1"
    local script="$2"
    local done_marker="$DONE_DIR/step_${step}"

    if [[ -f "$done_marker" ]]; then
        echo "  [skip] Step $step ($script) — already complete"
        return 0
    fi

    echo ""
    echo "================================================================"
    echo "  Step $step: $script"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
    local t0=$(date +%s)
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 "$script"
    local t1=$(date +%s)
    touch "$done_marker"
    echo ""
    echo "  Completed in $((t1 - t0)) seconds"
    echo "================================================================"
}

echo "========================================"
echo "  src_v6 Pipeline"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

TOTAL_START=$(date +%s)

for entry in "${STEPS[@]}"; do
    step="${entry%%:*}"
    script="${entry#*:}"
    if [[ "$step" > "$START" || "$step" == "$START" ]]; then
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
