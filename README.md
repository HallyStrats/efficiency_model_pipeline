# GPS-Only Energy Prediction Pipeline for Commercial Electric Motorcycles

This repository accompanies the paper *"GPS-Only Energy Prediction for a
Commercial Electric Motorcycle Fleet Using a Long Short-Term Memory Network"*
by H. Stratford and M. J. Booysen (Stellenbosch University). It contains the
full data-preprocessing, training, evaluation, and figure-generation pipeline
used to produce the reported results.

The model predicts per-trip energy consumption (Wh and Wh/km) for an electric
motorcycle (Roam Air) operating in a delivery and motorcycle-taxi fleet in
Nairobi, Kenya. **It uses only inputs that are recoverable from a standard GPS
trace** — latitude/longitude/timestamp, plus publicly available SRTM
elevation tiles and OpenStreetMap road classifications — so the same trained
model can in principle be applied to ICE motorcycle traces for route-matched
EV–ICE comparison.

## Headline results

Held-out test set: **7,245 trips from 25 riders withheld entirely from
training**, 3-seed LSTM ensemble, after a single global validation-set
bias-calibration scalar. All metrics timestep-weighted.

| Model                              | Wh MAE          | Wh R²            | Wh/km MAE       | Wh/km R²        |
|------------------------------------|-----------------|------------------|-----------------|-----------------|
| **GPS-only LSTM (3-seed ensemble)**| **22.0 ± 0.1**  | **0.949 ± 0.003**| **6.16 ± 0.04** | **0.325 ± 0.005**|
| Mean-Wh/km baseline (val-cal.)     | 31.54           | 0.884            | 7.84            | ≈ 0             |

Fleet-level aggregate energy error: **−2.2 %** across all 7,245 trips.

A constant-mean predictor receives the same one-parameter validation-set
calibration the LSTM receives, so any gap between the two reflects the
model's use of per-step features rather than asymmetric bias correction.

## Repository layout

```
src_v6/
├── 00_config.py                # All paths, thresholds, feature lists
├── 01_ingest.py                # Concatenate raw CSVs, assign rider IDs
├── 02_clean.py                 # GPS dropouts, BMS noise filtering
├── 03_features.py              # Kinematic features, SRTM elevation, target derivation
├── 04_segment.py               # Session + micro-trip segmentation
├── 05_split.py                 # Rider-disjoint train / val / test split
├── 06_train_gps.py             # Baseline GPS-only models (per-step XGBoost, trip XGBoost, LSTM)
├── 06b_osm_cache.py            # Build coordinate → OSM road-type cache via OSMnx
├── 06c_train_best.py           # Headline model: LSTM with 23 features, P90 Wh/km clip, 3 seeds
├── 06d_clip_sensitivity.py     # XGBoost P85/P90/P95/no-clip sensitivity probe
├── 06e_lstm_clip_sensitivity.py# Direct LSTM clip sensitivity (single seed, all other settings held constant)
├── 07_train_full.py            # Full-feature models (GPS + onboard sensors)
├── 08_evaluate.py              # Trip-level evaluation, fleet-level totals, model comparison plots
├── 09_tune.py                  # Optuna hyperparameter tuning for XGBoost
├── calibrated_metrics.py       # Paper-headline numbers + paper figures (val-calibrated, timestep-weighted)
├── run_pipeline.sh             # Orchestrator with resume markers (skips completed steps)
├── PIPELINE.md                 # Detailed per-step technical reference
├── requirements.txt
├── .gitignore
└── README.md                   # (this file)
```

## Quick start

### 1. Environment

```bash
git clone https://github.com/HallyStrats/efficiency_model_pipeline.git
cd efficiency_model_pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.11/3.12 on Apple Silicon (Darwin) and Linux. PyTorch is
used in CPU mode by default — the LSTM is small enough that CPU is fast
enough for both training and inference (full pipeline runs end-to-end in
≈ 3–4 h on an M-series Mac, with the Optuna XGBoost tune as the bottleneck).

### 2. Data

The raw fleet telemetry is **proprietary** to the operator and is not
included in this repository. The expected layout is:

```
efficiency_model/
├── all_data/                   # 26 raw CSV files (one per export window)
│   ├── 2023-mm-dd_export1.csv
│   └── …
└── src_v6/                     # this repo
```

Required raw columns per row: `imei_no`, `gps_date`, `lat`, `long`,
`altitude`, `vehicle_speed`, `ignition`, `odo_meter`, `rc` (remaining
charge in mAh), `c1v`–`c20v` (per-cell voltages in mV). Optional sensor
columns are picked up automatically if present (see `INGEST_COLS` in
`00_config.py`).

Once data is in place the pipeline writes all intermediate artefacts under
`src_v6/data/`, models under `src_v6/models/`, and results under
`src_v6/results_new/` — all of which are gitignored.

### 3. Run the pipeline

```bash
# Full end-to-end (resumes from the last completed step):
bash run_pipeline.sh

# Force re-run from a specific step onward:
bash run_pipeline.sh 06c           # retrain best model + everything after

# Clear all progress markers and start fresh:
bash run_pipeline.sh --clean
```

Each step writes a marker under `.pipeline_done/step_XX`; on restart, any
step whose marker exists is skipped. If a step is interrupted partway its
marker is never written, so it reruns cleanly.

### 4. Reproduce the paper headline numbers and figures

```bash
python3 calibrated_metrics.py
```

This loads the trained 3-seed LSTM ensemble from `models/gps_best/`, applies
the validation-set bias calibration, prints all timestep-weighted metrics in
both Wh and Wh/km space, and re-generates the three paper figures into
`../research_paper/draft/figures/` (path is hard-coded for the author's
working tree; adjust `FIG_DIR` at the top of the script if you have laid the
project out differently).

## Method in one paragraph

A GPS trace is enriched with SRTM 1-arcsecond elevation and OpenStreetMap
`highway` road-type labels. Per-step kinematic features (speed,
acceleration, jerk, heading, curvature, slope, time-of-day cyclic encoding,
and kinetic-energy/slope interactions) are computed, giving 23 input
features in total. A two-layer LSTM (hidden = 128, dropout = 0.2) predicts
the per-step energy intensity (Wh/km), trained with Huber loss on a target
clipped at the training-set 90th percentile (P90 = 69.9 Wh/km) to suppress
BMS coulomb-counting artefacts on short, low-energy steps. Trip-level
energy is recovered at inference by integrating per-step predictions over
step distance. Three independent seeds (42 / 123 / 456) are trained and
ensemble-averaged. A single global multiplicative scalar is fitted on the
rider-disjoint validation set to remove the small mean-bias introduced by
target clipping; the same calibration is applied symmetrically to the
constant-mean baseline.

For full technical detail see `PIPELINE.md` and the paper.

## Notes on reproducibility

- **Rider-disjoint evaluation.** The 25 test riders contribute no trips to
  training or validation, so reported metrics reflect generalisation to
  unseen operators, not unseen trips from seen operators.
- **Seed stability.** All headline numbers are reported as mean ± std
  across three independent training runs (seeds 42, 123, 456). Per-seed
  std is shown in the metrics output.
- **Timestep-weighted aggregation.** Every aggregate metric in the paper
  weights each trip by its number of GPS timesteps. Long trips dominate
  kWh demand; short trips dominate trip count. Weighting by step count
  matches energy-consumption importance and avoids inflating short-trip
  BMS quantisation noise into the headline scores.
- **OSM features cached.** Step `06b_osm_cache.py` downloads the Nairobi
  road network from OpenStreetMap once and pickles a coordinate → road-type
  lookup (4-decimal-place key, ≈ 10 m resolution). All downstream OSM
  feature lookups hit the cache.
- **SRTM cached.** Per-coordinate elevation lookups round to 4 decimal
  places for cache efficiency; the `srtm.py` library downloads tiles on
  demand on first use.

## Data availability

Fleet telemetry is proprietary to Roam Electric and cannot be shared
publicly. SRTM 1-arcsecond elevation tiles are public (U.S. Geological
Survey). OpenStreetMap data is public under the Open Database License.

## Citation

If you use this code or its results, please cite the accompanying paper
(see the paper for the BibTeX entry).

## Acknowledgements

The authors thank Roam Electric (Nairobi, Kenya) for providing the fleet
telemetry data that made this study possible.
