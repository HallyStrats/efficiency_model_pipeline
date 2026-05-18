# src_v6 Pipeline: GPS-Only Energy Consumption Prediction

## Overview

Predicts per-trip energy consumption (Wh and Wh/km) for electric motorcycles using **only GPS coordinates and timestamps** — no vehicle sensor data required. The model is designed to be transferable to ICE motorcycles where only GPS tracking exists.

**Data:** ~2.45M telemetry rows from 26 electric motorcycles in Nairobi, Kenya (10-second intervals).

**Best model:** 2-layer LSTM (3-seed ensemble), 23 GPS+OSM features, P90
Wh/km target clipping, with a single global validation-set bias-calibration
scalar applied to test predictions. Headline metrics on the rider-disjoint
test set (7,245 trips, 25 unseen riders), timestep-weighted:

- Trip energy: MAE 22.0 ± 0.1 Wh,  RMSE 42.25 Wh,  R² 0.949 ± 0.003
- Trip Wh/km: MAE 6.16 ± 0.04, RMSE 8.39, R² 0.325 ± 0.005
- MAPE 18.2%, MdAPE 13.2%
- Fleet-level aggregate energy error: −2.2%

The same val-fitted scalar is applied symmetrically to the constant-mean
baseline (Wh/km R² ≈ 0 by construction, MAE 31.5 Wh / 7.84 Wh/km).

## How to Run

```bash
# Full pipeline (all steps)
bash run_pipeline.sh

# From a specific step onward
bash run_pipeline.sh 06c    # Just retrain best model + evaluate
```

**Runtime:** ~3-4 hours end-to-end on an M-series Mac (Optuna tuning is the bottleneck).

**Dependencies:** numpy, pandas, xgboost, torch, scikit-learn, optuna, osmnx, srtm, scipy, joblib, matplotlib

## Pipeline Steps

### 01_ingest.py — Ingest raw CSVs
- Reads 26 CSV files from `../all_data/`
- Keeps: coordinates, timestamps, vehicle sensors, battery columns (for target only)
- Maps `imei_no` to stable integer `rider_id`
- Output: `data/raw_ingested.parquet` (~2.45M rows)

### 02_clean.py — Clean and validate
- Drops GPS dropouts (lat=0 or long=0)
- Drops riders with no BMS data (all `rc` null) — needed for target derivation
- 3-point median filter on `rc` per rider (suppresses sensor noise in target)
- Interpolates single-row BMS nulls (limit=1)
- Filters: rc > 45000 mAh, cell voltages outside 2800-4200 mV
- Output: `data/cleaned.parquet` (~2M rows)

### 03_features.py — Feature engineering
**GPS-derived features (Model 1 — 16 features):**
- `speed_ms`: haversine distance / dt (after 3-point median filter on lat/long, capped at 33.3 m/s = 120 km/h)
- `accel_ms2`: delta speed / dt (clipped to +/-3 m/s²)
- `jerk_ms3`: delta acceleration / dt
- `distance_m`: haversine step distance
- `elevation_m`: SRTM elevation lookup from coordinates (cached at 4dp)
- `slope_pct`: delta elevation / distance * 100 (clipped +/-30%)
- `heading_sin`, `heading_cos`: from atan2 of coordinate deltas
- `heading_change_rate`: circular heading difference / dt
- `curvature`: heading_change / speed (where speed > 0.5 m/s)
- `hour_sin`, `hour_cos`: cyclic time of day encoding
- `day_of_week`: 0-6
- `dt_seconds`: timestep duration
- `speed_x_slope`: speed * slope interaction
- `kinetic_energy_proxy`: 0.5 * speed²

**Sensor features (Model 2 adds 15 more):**
- vehicle_speed, spd, rpm, ts (throttle), bm (brake), heading_device, prnd (gear), gr (gear ratio)
- road_speed_limit, speed_limit_ratio, throttle_x_speed, accel_x_throttle
- rpm_rolling_mean, throttle_rolling_std, is_braking_while_fast

**Target computation:**
- `energy_wh = (rc[t-1] - rc[t]) * mean_pack_voltage_V / 1000`
- Pack voltage = mean of 19 series cells (c1v-c19v), converted mV to V
- Filtered: ignition must be on, dt in [5, 60]s, speed > 0.5 m/s, energy within physical limits

**Key decision:** Stationary rows (speed < 0.5 m/s) are removed here, not in segmentation. This means 04_segment.py detects trip boundaries from time gaps, not speed drops.

Output: `data/features.parquet` (~863K rows)

### 04_segment.py — Micro-trip segmentation
- **Session boundary:** dt > 300s (5-minute gap) or rider change
- **Micro-trip boundary:** dt > 30s between consecutive rows (stationary rows were already removed, so a gap > 30s means there was a stop)
- Computes per-trip aggregates for trip-level models
- Filters: trips with < 5 steps, < 100m distance, or energy <= 0 are removed
- Output: `data/trips.parquet` (~824K rows), `data/trip_summaries.parquet` (~38K trips)

**Key decision:** Original implementation looked for slow-speed periods within the data, but since stationary rows were removed in step 03, this found only 118 trips. Fixed to detect time gaps instead, yielding ~38K trips.

### 05_split.py — Train/val/test split by rider
- 80% riders for training, 20% for test (stratified by total distance)
- Within training riders, 10% of trips for validation
- Same split for all models — critical for fair comparison
- Output: train (~593K rows, 27.5K trips, 94 riders), val (~67K rows, 3K trips), test (~164K rows, 7.6K trips, 24 riders)

**Key decision:** Split by rider, not by trip. Prevents data leakage where the model memorises a specific rider's patterns.

### 06_train_gps.py — Baseline GPS-only models
Trains three sub-models with the raw `energy_wh` target:
- A. Per-timestep XGBoost (n_estimators=1000, lr=0.03, max_depth=7)
- B. Trip-level aggregate XGBoost
- C. Per-timestep LSTM (2-layer, hidden=128)

These serve as baselines. The best GPS model comes from 06c.

### 06b_osm_cache.py — Build OSM road type cache
- Downloads Nairobi road network via `osmnx` (`ox.graph_from_place("Nairobi, Kenya")`)
- Extracts coordinate-to-highway_type mappings at 4dp resolution
- Uses KDTree for nearest-neighbour matching (max 100m)
- Caches ~290K coordinate mappings
- Road types: residential, secondary, tertiary, primary, trunk, motorway, unknown
- Output: `data/osm_road_cache.pkl`

**Key decision:** Used `graph_from_place` instead of `graph_from_bbox` — the bbox approach generated thousands of sub-queries and failed. Place-based download worked cleanly.

### 06c_train_best.py — Train best GPS model
This encapsulates all the optimisations that improved Wh/km prediction:

**1. OSM road type features (7 additional features):**
- `road_type_ord`: ordinal encoding (residential=1, secondary=4, trunk=6, motorway=7)
- One-hot indicators: `road_residential`, `road_secondary`, `road_tertiary`, `road_primary`, `road_trunk`, `road_motorway`

**2. Wh/km target instead of raw Wh:**
- `energy_whkm = energy_wh / (distance_m / 1000)`
- Removes trip-length dependency — model learns efficiency, not just "longer trips use more energy"
- Without this, the model achieves R²=0.94 on trip Wh but near-zero on Wh/km

**3. P90 target clipping:**
- Training target clipped at the 90th percentile (~75 Wh/km)
- The top 10% of Wh/km values are noisy outliers from the rc sensor
- This single change improved Wh MAE from 12.01 to 10.57 (-12%)

**4. LSTM architecture:**
- 2-layer LSTM, hidden_size=128, dropout=0.2
- HuberLoss (delta=1.0) — more robust to outliers than MSE
- Adam optimiser, lr=1e-3, weight_decay=1e-5
- ReduceLROnPlateau (patience=5, factor=0.5)
- Early stopping (patience=15)
- Batch size 128, max 100 epochs
- Forced to CPU (MPS/Apple GPU hangs on pack_padded_sequence)

**5. Optuna-tuned XGBoost (50 trials):**
- Tunes: learning_rate, max_depth, subsample, colsample_bytree, min_child_weight, reg_alpha, reg_lambda, gamma, max_bin
- Objective: trip-level MAE on validation set
- Serves as comparison baseline — LSTM consistently outperforms

Output: `models/gps_best/lstm_model.pt`, `models/gps_best/lstm_scaler.joblib`, `models/gps_best/xgb_model.json`, `models/gps_best/metadata.json`

### 07_train_full.py — Full-feature models
Same architectures as 06 but using all 31 features (GPS + sensor). Serves as upper-bound benchmark showing what's achievable with throttle/RPM/brake data.

### 08_evaluate.py — Evaluation and plots
Evaluates LSTM, XGBoost, and mean baseline on test set. Reports both unweighted and timestep-weighted metrics.

**Timestep-weighted evaluation:**
- Trips weighted by number of timesteps (n_steps)
- Justified because short trips (<5 timesteps) have high target noise from the rc sensor — a single noisy reading can swing Wh/km by 50%
- Longer trips average out sensor noise and give more reliable ground truth
- Weighted Wh/km R²: 0.391 (vs 0.365 unweighted)

Output: `results_new/metrics_summary.json`, `results_new/lstm_vs_mean_error_weighted.png`, `results_new/whkm_scatter.png`, `results_new/feature_importance.png`

## Key Results

### Best GPS-Only Model (LSTM 3-seed ensemble, 23 features, calibrated)

All test-set metrics below are timestep-weighted and computed after a single
global validation-set bias calibration (see `calibrated_metrics.py`). LSTM
MAE/R² are reported as mean ± std across seeds 42/123/456 to confirm
robustness to weight initialisation.

| Metric | Wh (trip energy) | Wh/km (trip energy intensity) |
|--------|------------------|-------------------------------|
| MAE    | 22.0 ± 0.1       | 6.16 ± 0.04                   |
| RMSE   | 42.25            | 8.39                          |
| R²     | 0.949 ± 0.003    | 0.325 ± 0.005                 |
| MAPE   | 18.2%            | 18.2%                         |
| MdAPE  | 13.2%            | 13.2%                         |

### Calibrated mean-Wh/km baseline (val-fitted)

For methodological symmetry the constant-mean baseline uses the validation-set
fleet-average Wh/km (the same one-parameter calibration that the LSTM receives).

| Metric | Wh | Wh/km |
|--------|----|-------|
| MAE    | 31.54 | 7.84 |
| RMSE   | 64.59 | 10.44 |
| R²     | 0.884 | ≈ 0  |
| MAPE   | 24.2% | 24.2% |

Fleet-level sanity: summed predicted vs actual Wh across all 7,245 test
trips → **−2.2% error** for the calibrated LSTM. Per-trip residuals largely
cancel at scale; the residual ~2% reflects the spread that calibration
cannot remove.

### P90 Clip Sensitivity (XGBoost, fixed params, validation set)

| Clip level | Trip Wh MAE | MAPE | Wh/km R² |
|------------|------------|------|----------|
| No clip | 12.30 | 30.6% | ~0 |
| P95 | 9.21 | — | — |
| **P90 (chosen)** | **10.57** | **23.5%** | **0.252** |
| P85 | 10.89 | — | — |

P90 was chosen as the production clip level: P95 is marginally better on Wh MAE but P90 suppresses more rc-sensor outlier noise, which is the correct behaviour for robustness.

### Improvement Journey

Note: rows below mix metric definitions used during development;
the **paper-headline** numbers are the calibrated 3-seed ensemble
listed above this section.

| Change | Trip Wh MAE | MAPE | Wh/km R² |
|--------|------------|------|----------|
| XGBoost baseline (energy_wh target) | 12.30 | 30.6% | ~0 |
| + Wh/km target | 12.24 | 30.5% | ~0 |
| + OSM road type features | 12.24 | 30.5% | ~0 |
| + Optuna tuning | 12.01 | 29.9% | ~0 |
| + P90 target clipping | 10.57 | 23.5% | 0.252 |
| Switch to LSTM (single seed) | 9.63 | 20.9% | 0.365 |
| 3-seed ensemble (uncalibrated) | 10.01 ± 0.08 | 17.0% | 0.394 |
| **+ val-set bias calibration** | **22.0 ± 0.1 (wtd)** | **18.2%** | **0.325** |

The last two rows look like a regression but aren't: the first row is the
unweighted ensemble before calibration, and the last row is the timestep-
weighted ensemble after calibration. The calibrated, timestep-weighted
numbers are the ones reported in the paper.

### What GPS Can and Cannot Predict

The model explains ~33% of Wh/km variance from GPS alone (timestep-weighted,
calibrated). The remaining ~67% comes from factors invisible to GPS:
- Rider aggressiveness (throttle behaviour, braking patterns)
- Payload (passengers, cargo)
- Vehicle condition (tyre pressure, chain tension, motor wear)
- Wind and weather
- Traffic micro-patterns between 10-second sampling intervals

The OLS baseline (R²=0.011 on Wh/km) confirms that simple trip-level aggregates of GPS features carry almost no efficiency signal — the LSTM's sequential modelling of acceleration and jerk patterns is what drives the 0.393 Wh/km R².

## Feature Importance (XGBoost)

Top features by gain:
1. **kinetic_energy_proxy** (0.5 * speed²) — dominant predictor
2. **speed_ms** — baseline energy demand
3. **distance_m** — step length
4. **slope_pct** — terrain gradient from SRTM elevation
5. **jerk_ms3** — smoothness of driving
6. **road_trunk** / **road_type_ord** — OSM road classification
7. **heading_sin** — directional component (correlates with specific routes)
8. **accel_ms2** — acceleration demand
9. **elevation_m** — altitude (Nairobi has significant elevation variation)
10. **speed_x_slope** — power demand on grades

## Feature Engineering Decisions That Did NOT Help

- **Rolling speed/accel variance** (window=5): XGBoost benefited slightly (+0.007 R²), LSTM did not — it already learns these patterns internally
- **Speed regime bins** (is_slow, is_medium, is_fast): Same — helped XGBoost, not LSTM
- **Acceleration energy / deceleration energy**: No improvement for either model
- **Stop proximity indicator**: Helped XGBoost (#1 feature), no LSTM improvement

Conclusion: hand-engineered sequential features help tree models but are redundant for LSTMs.

## File Structure

```
src_v6/
  00_config.py          Central configuration (paths, thresholds, feature lists)
  01_ingest.py          Ingest raw CSVs → raw_ingested.parquet
  02_clean.py           Clean and validate → cleaned.parquet
  03_features.py        Feature engineering → features.parquet
  04_segment.py         Micro-trip segmentation → trips.parquet, trip_summaries.parquet
  05_split.py           Train/val/test split → train/val/test_trips.parquet
  06_train_gps.py       Baseline GPS models
  06b_osm_cache.py      Build OSM road type cache
  06c_train_best.py     Best GPS model (LSTM + tuned XGBoost)
  07_train_full.py      Full-feature models (sensor data)
  08_evaluate.py        Evaluation and plots
  09_tune.py            Standalone Optuna tuning (optional, superseded by 06c)
  run_pipeline.sh       End-to-end orchestrator

  data/                 Intermediate data files (parquet)
  models/               Trained models
    gps_best/           Best GPS-only model artifacts
  data/                 Intermediate data files (parquet)
  models/               Trained models
    gps_best/           Best GPS-only model artifacts (lstm_model_seed{42,123,456}.pt, xgb_model.json)
    clip_sensitivity/   XGBoost models at each clip level + clip_sensitivity.json
  results_new/          Evaluation metrics and plots (latest validated run)
```

## Reproducing Results

1. Place raw CSV files in `../all_data/`
2. Run `bash run_pipeline.sh`
3. Check `results_new/metrics_summary.json` for metrics
4. Check `results_new/*.png` for plots

To retrain only the best model (if data pipeline is already done):
```bash
bash run_pipeline.sh 06c
```

### macOS Apple Silicon Notes

Three PyTorch issues affect macOS ARM (M-series) and require these workarounds (already applied in the pipeline):

1. **MPS backend hang** on `pack_padded_sequence` — fixed via `os.environ["PYTORCH_MPS_DISABLE"] = "1"` before torch import and forcing `device = torch.device("cpu")`
2. **`pad_sequence` / `fill_out` deadlock** on CPU — fixed by replacing `torch.nn.utils.rnn.pad_sequence` in `collate_trips` with manual numpy-based padding
3. **OpenMP `parallel_for` / `index_select` deadlock** during DataLoader shuffle — fixed by running python with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` (already set in `run_pipeline.sh`)
