"""
06d_clip_sensitivity.py

Validates the P90 Wh/km target-clip design choice by training XGBoost with
four clip levels on the same feature set as 06c, then evaluating on the
validation set. Fixed XGBoost params are used across all levels so the
comparison is fair (isolates the clip effect, not the tuning effect).

Output:
  models/clip_sensitivity/clip_sensitivity.json   — results table
  models/clip_sensitivity/xgb_{label}.json        — model per level
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

from importlib import import_module

cfg = import_module("00_config")
train_best = import_module("06c_train_best")

CLIP_LEVELS = [
    ("no_clip", None),
    ("p95",     0.95),
    ("p90",     0.90),
    ("p85",     0.85),
]

XGB_PARAMS = {
    "n_estimators":        1000,
    "learning_rate":       0.03,
    "max_depth":           7,
    "subsample":           0.8,
    "colsample_bytree":    0.8,
    "min_child_weight":    10,
    "reg_alpha":           0.1,
    "reg_lambda":          1.0,
    "tree_method":         "hist",
    "random_state":        cfg.SEED,
    "early_stopping_rounds": 30,
}


def main():
    out_dir = os.path.join(cfg.MODEL_DIR, "clip_sensitivity")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "clip_sensitivity.json")
    if os.path.exists(out_path):
        print(f"Clip sensitivity already complete — skipping ({out_path})")
        return

    with open(os.path.join(cfg.DATA_DIR, "osm_road_cache.pkl"), "rb") as f:
        osm_cache = pickle.load(f)

    train = train_best.add_road_features(
        pd.read_parquet(os.path.join(cfg.DATA_DIR, "train_trips.parquet")), osm_cache,
    )
    val = train_best.add_road_features(
        pd.read_parquet(os.path.join(cfg.DATA_DIR, "val_trips.parquet")), osm_cache,
    )

    gps_features = [f for f in cfg.GPS_FEATURES if f in train.columns and train[f].notna().any()]
    feature_cols = gps_features + train_best.ROAD_FEATURE_COLS

    for df in [train, val]:
        df["energy_whkm"] = np.where(
            df["distance_m"] > 0,
            df["energy_wh"] / (df["distance_m"] / 1000.0),
            np.nan,
        )

    # Precompute validation actuals (unclipped) for evaluation
    val_valid = val["energy_whkm"].notna() & np.isfinite(val["energy_whkm"])
    val_trip_wh = val[val_valid].groupby("trip_id")["energy_wh"].sum()
    val_trip_dist = val[val_valid].groupby("trip_id")["distance_m"].sum()
    val_trip_ids = val.loc[val_valid, "trip_id"].values
    val_X = val.loc[val_valid, feature_cols].fillna(0).values
    val_dists = val.loc[val_valid, "distance_m"].values

    results = []
    print(f"\n{'Clip level':<10} {'Clip value':>12} {'Val Wh MAE':>12} {'MAPE':>8} {'Wh/km R²':>10}")
    print("-" * 58)

    for label, quantile in CLIP_LEVELS:
        raw_train = train["energy_whkm"]
        if quantile is not None:
            clip_val = float(raw_train.quantile(quantile))
            train_target = raw_train.clip(upper=clip_val)
            val_target = val["energy_whkm"].clip(upper=clip_val)
        else:
            clip_val = None
            train_target = raw_train
            val_target = val["energy_whkm"]

        tr_valid = train_target.notna() & np.isfinite(train_target)
        X_tr = train.loc[tr_valid, feature_cols].fillna(0).values
        y_tr = train_target[tr_valid].values

        va_valid = val_target.notna() & np.isfinite(val_target)
        X_va = val.loc[va_valid, feature_cols].fillna(0).values
        y_va = val_target[va_valid].values

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=0)
        model.save_model(os.path.join(out_dir, f"xgb_{label}.json"))

        # Evaluate at trip level against unclipped actuals
        preds_whkm = model.predict(val_X)
        pred_wh_ts = preds_whkm * (val_dists / 1000.0)
        pred_df = pd.DataFrame({"trip_id": val_trip_ids, "pred_wh": pred_wh_ts})
        pred_per_trip = pred_df.groupby("trip_id")["pred_wh"].sum()

        common = val_trip_wh.index.intersection(pred_per_trip.index)
        act_wh = val_trip_wh[common].values
        pred_wh = pred_per_trip[common].values
        act_dist = val_trip_dist[common].values

        mae = mean_absolute_error(act_wh, pred_wh)
        mape = np.mean(np.abs((act_wh - pred_wh) / np.clip(act_wh, 0.01, None))) * 100

        act_whkm = act_wh / (act_dist / 1000.0)
        pred_whkm = pred_wh / (act_dist / 1000.0)
        mask = np.isfinite(act_whkm) & np.isfinite(pred_whkm) & (act_whkm > 0.1) & (act_whkm < 200)
        r2_whkm = r2_score(act_whkm[mask], pred_whkm[mask])

        clip_str = f"{clip_val:.1f}" if clip_val is not None else "none"
        marker = " ←" if label == "p90" else ""
        print(f"{label:<10} {clip_str:>12} {mae:>12.2f} {mape:>7.1f}% {r2_whkm:>10.3f}{marker}")

        results.append({
            "label": label,
            "quantile": quantile,
            "clip_value_whkm": clip_val,
            "val_wh_mae": float(mae),
            "val_mape_pct": float(mape),
            "val_whkm_r2": float(r2_whkm),
            "n_val_trips": int(len(common)),
        })

    out_path = os.path.join(out_dir, "clip_sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
