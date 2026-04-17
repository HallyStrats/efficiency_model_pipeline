"""
07_train_full.py

Trains Model 2: Full-Feature models using all allowed (non-EV-specific)
features — GPS-derived + sensor features (throttle, brake, RPM, etc.).

Three sub-models:
  A. Per-timestep XGBoost → aggregate to trip
  B. Trip-level aggregate XGBoost
  C. Per-timestep LSTM → aggregate to trip

Output:
  models/full_xgb_timestep.json
  models/full_xgb_trip.json
  models/full_lstm.pt
  models/full_scaler.joblib
"""

import os
import json

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from importlib import import_module
cfg = import_module("00_config")

# Reuse LSTM infrastructure from 06_train_gps
train_gps = import_module("06_train_gps")


def main():
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)

    # Load data
    train_trips = pd.read_parquet(os.path.join(cfg.DATA_DIR, "train_trips.parquet"))
    val_trips = pd.read_parquet(os.path.join(cfg.DATA_DIR, "val_trips.parquet"))
    train_sum = pd.read_parquet(os.path.join(cfg.DATA_DIR, "train_summaries.parquet"))
    val_sum = pd.read_parquet(os.path.join(cfg.DATA_DIR, "val_summaries.parquet"))

    # Determine available full features (GPS + sensor)
    full_features = train_gps.get_available_features(train_trips, cfg.FULL_FEATURES)
    print(f"Full features available: {len(full_features)}/{len(cfg.FULL_FEATURES)}")
    for f in full_features:
        pct = train_trips[f].notna().mean() * 100
        print(f"  {f:<30} {pct:5.1f}% non-null")

    # Full trip-level feature columns (all trip_ columns except target/ID)
    full_trip_features = [c for c in train_sum.columns
                          if c.startswith("trip_") and c not in
                          ("trip_id", "trip_energy_wh", "trip_wh_per_km", "rider_id")]

    # --- Model A: Per-timestep XGBoost ---
    train_gps.train_xgb_timestep(
        train_trips, val_trips, full_features,
        os.path.join(cfg.MODEL_DIR, "full_xgb_timestep.json"),
    )

    # --- Model B: Trip-level XGBoost ---
    train_gps.train_xgb_trip(
        train_sum, val_sum, full_trip_features,
        os.path.join(cfg.MODEL_DIR, "full_xgb_trip.json"),
    )

    # --- Model C: LSTM ---
    train_gps.train_lstm(
        train_trips, val_trips, full_features,
        os.path.join(cfg.MODEL_DIR, "full_lstm.pt"),
        os.path.join(cfg.MODEL_DIR, "full_scaler.joblib"),
    )

    print("\n" + "="*70)
    print("Full-feature model training complete.")
    print("="*70)


if __name__ == "__main__":
    main()
