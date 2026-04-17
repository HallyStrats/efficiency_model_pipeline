"""
00_config.py

Central configuration for the src_v6 pipeline.
All paths, thresholds, column lists, and feature definitions imported by every script.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SRC_DIR, "..")
RAW_DIR     = os.path.join(PROJECT_DIR, "all_data")
DATA_DIR    = os.path.join(SRC_DIR, "data")
MODEL_DIR   = os.path.join(SRC_DIR, "models")
RESULTS_DIR = os.path.join(SRC_DIR, "results")

# ---------------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Ingestion — columns to keep from raw CSVs
# ---------------------------------------------------------------------------
INGEST_COLS = [
    # Identifiers & timestamps
    "imei_no", "gps_date", "lat", "long",
    # Vehicle state
    "is_ignition_on", "vehicle_speed", "spd", "heading", "altitude",
    # Motor / drivetrain (allowed for Model 2)
    "rpm", "ts", "bm", "prnd", "gr",
    # Context
    "road_speed_limit", "sat", "rssi", "odometer", "engine_hours",
    # Battery / energy — target derivation only (never used as input features)
    "rc", "fc",
    "c1v", "c2v", "c3v", "c4v", "c5v", "c6v", "c7v", "c8v", "c9v", "c10v",
    "c11v", "c12v", "c13v", "c14v", "c15v", "c16v", "c17v", "c18v", "c19v", "c20v",
]

CELL_COLS = [f"c{i}v" for i in range(1, 21)]

# ---------------------------------------------------------------------------
# Cleaning thresholds
# ---------------------------------------------------------------------------
RC_MAX_MAH          = 45_000       # max plausible remaining charge
CELL_V_MIN_MV       = 2_800        # min cell voltage
CELL_V_MAX_MV       = 4_200        # max cell voltage
RC_MEDIAN_WINDOW    = 3            # median filter window for rc noise

# ---------------------------------------------------------------------------
# Feature engineering thresholds
# ---------------------------------------------------------------------------
DT_MIN_S            = 5            # min timestep to keep
DT_MAX_S            = 60           # max timestep to keep
SESSION_GAP_S       = 300          # 5 min → session boundary
SPEED_CAP_MS        = 33.33        # 120 km/h cap on derived speed
ACCEL_CAP_MS2       = 3.0          # ±3 m/s² clip
SLOPE_CAP_PCT       = 30.0         # ±30% clip
SRTM_ROUND_DP       = 4            # coordinate rounding for SRTM cache (~10 m)
MIN_SPEED_MS        = 0.5          # below this → stationary
ELEVATION_MIN_M     = 1_000        # Nairobi basin floor

# Energy target limits
DISCHARGE_MAX_W     = 6_000        # max plausible discharge power (watts)
REGEN_MAX_W         = 3_000        # max plausible regen power (watts)
RC_STEP_MAX_MAH     = 500          # max plausible rc change per step
N_SERIES_CELLS      = 19           # cells in series for pack voltage
NOMINAL_CELL_MV     = 3_600        # nominal cell voltage (mV)

# ---------------------------------------------------------------------------
# Trip segmentation
# ---------------------------------------------------------------------------
MICROTRIP_GAP_S     = 30           # stationary gap to split micro-trips
MIN_TRIP_STEPS      = 5            # min timesteps per trip
MIN_TRIP_DIST_M     = 100          # min distance per trip

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_RIDER_FRAC     = 0.20
VAL_TRIP_FRAC       = 0.10

# ---------------------------------------------------------------------------
# Feature column lists
# ---------------------------------------------------------------------------

# GPS-derived features (Model 1) — derivable from lat/long/timestamp only
GPS_FEATURES = [
    "speed_ms",
    "accel_ms2",
    "jerk_ms3",
    "distance_m",
    "elevation_m",
    "slope_pct",
    "heading_sin",
    "heading_cos",
    "heading_change_rate",
    "curvature",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "dt_seconds",
    "speed_x_slope",
    "kinetic_energy_proxy",
]

# Additional sensor features (Model 2 gets GPS_FEATURES + these)
SENSOR_FEATURES = [
    "vehicle_speed",
    "spd",
    "rpm",
    "ts",
    "bm",
    "heading_device",
    "prnd",
    "gr",
    "road_speed_limit",
    "speed_limit_ratio",
    "throttle_x_speed",
    "accel_x_throttle",
    "rpm_rolling_mean",
    "throttle_rolling_std",
    "is_braking_while_fast",
]

FULL_FEATURES = GPS_FEATURES + SENSOR_FEATURES

# Target column
TARGET_COL = "energy_wh"

# Columns excluded from features (EV-specific — used only for target derivation)
EV_ONLY_COLS = [
    "rc", "fc", "curr", "tv",
    "bats", "ps", "dc", "rc_p",
    "bv_l", "bv_h", "bc_l", "bc_h",
] + CELL_COLS

# ---------------------------------------------------------------------------
# Model hyperparameters (defaults — tuning overrides these)
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    "n_estimators":         1000,
    "learning_rate":        0.03,
    "max_depth":            7,
    "subsample":            0.8,
    "colsample_bytree":     0.8,
    "min_child_weight":     10,
    "reg_alpha":            0.1,
    "reg_lambda":           1.0,
    "tree_method":          "hist",
    "random_state":         SEED,
}
XGB_EARLY_STOPPING = 30

LSTM_PARAMS = {
    "hidden_size":          128,
    "num_layers":           2,
    "dropout":              0.2,
    "learning_rate":        1e-3,
    "weight_decay":         1e-5,
    "batch_size":           64,
    "max_epochs":           100,
    "patience":             15,
}
