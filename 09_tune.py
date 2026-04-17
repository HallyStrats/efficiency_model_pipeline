"""
09_tune.py

Hyperparameter tuning with Optuna for the best-performing XGBoost models
from each track (GPS-only and Full-feature).

Tunes:
  - XGBoost hyperparameters (max_depth, learning_rate, subsample, etc.)
  - Optimizes trip-level MAE on validation set

Output:
  models/gps_best_params.json
  models/full_best_params.json
  models/optuna_study.db
"""

import json
import os

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from importlib import import_module
cfg = import_module("00_config")
train_gps = import_module("06_train_gps")


def create_objective(train_trips, val_trips, feature_cols, target_col, model_type="timestep"):
    """Create an Optuna objective for XGBoost tuning."""

    if model_type == "timestep":
        X_train, y_train, X_val, y_val = train_gps.prepare_timestep_data(
            train_trips, val_trips, feature_cols, target_col
        )
        val_trip_ids = val_trips["trip_id"]
        val_actual_per_trip = val_trips.groupby("trip_id")[target_col].sum()
    else:
        X_train = train_trips[feature_cols].fillna(0).values
        y_train = train_trips["trip_energy_wh"].values
        X_val = val_trips[feature_cols].fillna(0).values
        y_val = val_trips["trip_energy_wh"].values
        valid_t = ~np.isnan(y_train)
        valid_v = ~np.isnan(y_val)
        X_train, y_train = X_train[valid_t], y_train[valid_t]
        X_val, y_val = X_val[valid_v], y_val[valid_v]

    def objective(trial):
        params = {
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 30),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "tree_method": "hist",
            "random_state": cfg.SEED,
            "early_stopping_rounds": cfg.XGB_EARLY_STOPPING,
        }

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

        if model_type == "timestep":
            # Predict per-timestep, aggregate to trips
            valid_mask = val_trips[target_col].notna()
            preds_all = model.predict(val_trips.loc[valid_mask, feature_cols].fillna(0).values)
            pred_df = pd.DataFrame({
                "trip_id": val_trips.loc[valid_mask, "trip_id"].values,
                "pred": preds_all,
            })
            pred_per_trip = pred_df.groupby("trip_id")["pred"].sum()
            common = val_actual_per_trip.index.intersection(pred_per_trip.index)
            mae = mean_absolute_error(val_actual_per_trip[common], pred_per_trip[common])
        else:
            preds = model.predict(X_val)
            mae = mean_absolute_error(y_val, preds)

        return mae

    return objective


def main():
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)

    train_trips = pd.read_parquet(os.path.join(cfg.DATA_DIR, "train_trips.parquet"))
    val_trips = pd.read_parquet(os.path.join(cfg.DATA_DIR, "val_trips.parquet"))
    train_sum = pd.read_parquet(os.path.join(cfg.DATA_DIR, "train_summaries.parquet"))
    val_sum = pd.read_parquet(os.path.join(cfg.DATA_DIR, "val_summaries.parquet"))

    gps_features = train_gps.get_available_features(train_trips, cfg.GPS_FEATURES)
    full_features = train_gps.get_available_features(train_trips, cfg.FULL_FEATURES)

    storage = f"sqlite:///{os.path.join(cfg.MODEL_DIR, 'optuna_study.db')}"
    n_trials = 40

    # --- Tune GPS-only XGBoost timestep ---
    print("="*70)
    print("Tuning GPS-Only XGBoost Timestep")
    print("="*70)
    study_gps = optuna.create_study(
        study_name="gps_xgb_timestep",
        direction="minimize",
        storage=storage,
        load_if_exists=True,
    )
    objective_gps = create_objective(train_trips, val_trips, gps_features, cfg.TARGET_COL, "timestep")
    study_gps.optimize(objective_gps, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest GPS trial: MAE={study_gps.best_value:.4f}")
    print(f"  Params: {study_gps.best_params}")

    gps_params_path = os.path.join(cfg.MODEL_DIR, "gps_best_params.json")
    with open(gps_params_path, "w") as f:
        json.dump(study_gps.best_params, f, indent=2)

    # --- Tune Full-feature XGBoost timestep ---
    print("\n" + "="*70)
    print("Tuning Full-Feature XGBoost Timestep")
    print("="*70)
    study_full = optuna.create_study(
        study_name="full_xgb_timestep",
        direction="minimize",
        storage=storage,
        load_if_exists=True,
    )
    objective_full = create_objective(train_trips, val_trips, full_features, cfg.TARGET_COL, "timestep")
    study_full.optimize(objective_full, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest Full trial: MAE={study_full.best_value:.4f}")
    print(f"  Params: {study_full.best_params}")

    full_params_path = os.path.join(cfg.MODEL_DIR, "full_best_params.json")
    with open(full_params_path, "w") as f:
        json.dump(study_full.best_params, f, indent=2)

    # --- Retrain best models ---
    print("\n" + "="*70)
    print("Retraining with best parameters")
    print("="*70)

    # GPS best
    best_gps = {**cfg.XGB_PARAMS, **study_gps.best_params}
    model_gps = xgb.XGBRegressor(**best_gps, early_stopping_rounds=cfg.XGB_EARLY_STOPPING)
    X_tr, y_tr, X_v, y_v = train_gps.prepare_timestep_data(
        train_trips, val_trips, gps_features, cfg.TARGET_COL
    )
    model_gps.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=50)
    model_gps.save_model(os.path.join(cfg.MODEL_DIR, "gps_xgb_timestep_tuned.json"))
    print("  Saved tuned GPS model")

    # Full best
    best_full = {**cfg.XGB_PARAMS, **study_full.best_params}
    model_full = xgb.XGBRegressor(**best_full, early_stopping_rounds=cfg.XGB_EARLY_STOPPING)
    X_tr, y_tr, X_v, y_v = train_gps.prepare_timestep_data(
        train_trips, val_trips, full_features, cfg.TARGET_COL
    )
    model_full.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=50)
    model_full.save_model(os.path.join(cfg.MODEL_DIR, "full_xgb_timestep_tuned.json"))
    print("  Saved tuned Full model")

    print("\nTuning complete.")


if __name__ == "__main__":
    main()
