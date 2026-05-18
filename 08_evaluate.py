"""
08_evaluate.py

Evaluates GPS-only models on the held-out test set.

Models / baselines evaluated:
  1. LSTM (multi-seed ensemble, mean ± std across 3 seeds)
  2. Optuna-tuned XGBoost
  3. OLS baseline (trip-level: distance_km + elevation_gain_m + mean_speed_ms → Wh)
  4. Constant mean Wh/km baseline

Additional validation output:
  - Fleet-level total energy sanity check (sum predicted vs actual Wh)
  - P90 clip sensitivity table (loaded from models/clip_sensitivity/ if present)

Metrics:
  - Unweighted (all trips equal)
  - Timestep-weighted (longer trips weighted by n_steps — justified because
    short trips have higher target noise from rc sensor)

Output:
  results/metrics_summary.json
  results/lstm_vs_mean_error_weighted.png
  results/feature_importance.png
  results/whkm_scatter.png
"""

import json
import os
import pickle

os.environ["PYTORCH_MPS_DISABLE"] = "1"  # must be set before torch import

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from importlib import import_module

cfg = import_module("00_config")
train_best = import_module("06c_train_best")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def weighted_metrics(actual, pred, weights):
    """Timestep-weighted regression metrics."""
    err = actual - pred
    abs_err = np.abs(err)
    pct_err = np.abs(err / np.clip(actual, 0.01, None))

    wmae  = np.average(abs_err, weights=weights)
    wrmse = np.sqrt(np.average(err**2, weights=weights))
    wmape = np.average(pct_err, weights=weights) * 100

    ss_res = np.average(err**2, weights=weights)
    w_mean = np.average(actual, weights=weights)
    ss_tot = np.average((actual - w_mean)**2, weights=weights)
    wr2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    sort_idx = np.argsort(pct_err)
    cum_w = np.cumsum(weights[sort_idx])
    mid = np.searchsorted(cum_w, cum_w[-1] / 2)
    wmdape = pct_err[sort_idx[min(mid, len(sort_idx) - 1)]] * 100

    bias = np.average(err, weights=weights)
    return {"wmae": wmae, "wrmse": wrmse, "wr2": wr2,
            "wmape": wmape, "wmdape": wmdape, "bias": bias}


def unweighted_metrics(actual, pred):
    """Standard regression metrics."""
    return {
        "mae":   mean_absolute_error(actual, pred),
        "rmse":  np.sqrt(mean_squared_error(actual, pred)),
        "r2":    r2_score(actual, pred),
        "mape":  np.mean(np.abs((actual - pred) / np.clip(actual, 0.01, None))) * 100,
        "mdape": np.median(np.abs((actual - pred) / np.clip(actual, 0.01, None))) * 100,
        "bias":  float((pred - actual).mean()),
    }


# ---------------------------------------------------------------------------
# LSTM inference
# ---------------------------------------------------------------------------

def predict_lstm_from_path(model_path, scaler_path, test_df, feature_cols):
    """Load a specific LSTM checkpoint and return per-trip prediction lists."""
    scaler = joblib.load(scaler_path)
    test_sc = test_df.copy()
    test_sc[feature_cols] = scaler.transform(test_df[feature_cols].fillna(0))

    test_ds = train_best.TripDataset(test_sc, feature_cols, "energy_whkm")

    model = train_best.TripLSTM(input_size=len(feature_cols))
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    all_preds = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            seq, _ = test_ds[i]
            length = torch.tensor([len(test_ds.sequences[i])])
            pred = model(seq.unsqueeze(0), length).numpy().flatten()[:len(test_ds.sequences[i])]
            all_preds.append(pred)
    return all_preds


def build_trip_results(test_df, all_preds, pred_col="lstm_whkm"):
    """Aggregate per-timestep Wh/km predictions to trip-level totals."""
    valid = test_df["energy_whkm"].notna() & np.isfinite(test_df["energy_whkm"])
    tv = test_df[valid]
    ids = tv["trip_id"].values
    breaks = np.where(np.diff(ids) != 0)[0] + 1
    starts = np.concatenate([[0], breaks])
    ends   = np.concatenate([breaks, [len(ids)]])

    rows = []
    for idx, (s, e) in enumerate(zip(starts, ends)):
        if idx < len(all_preds) and e - s > 0:
            c = tv.iloc[s:e]
            rows.append({
                "trip_id":   c["trip_id"].iloc[0],
                pred_col:    float(np.mean(all_preds[idx][:e - s])),
                "actual_wh": float(c["energy_wh"].sum()),
                "total_dist": float(c["distance_m"].sum()),
                "n_steps":   int(e - s),
            })

    tr = pd.DataFrame(rows)
    tr["actual_whkm"] = tr["actual_wh"] / (tr["total_dist"] / 1000.0)
    tr["pred_wh"] = tr[pred_col] * (tr["total_dist"] / 1000.0)
    return tr[tr["actual_whkm"].between(0.1, 200) & np.isfinite(tr["actual_whkm"])].copy()


# ---------------------------------------------------------------------------
# OLS baseline helpers
# ---------------------------------------------------------------------------

def compute_ols_trip_features(df):
    """Compute trip-level feature matrix for OLS baseline."""
    def elev_gain(series):
        vals = series.values
        diffs = np.diff(vals)
        return float(np.sum(diffs[diffs > 0])) if len(diffs) > 0 else 0.0

    agg = df.groupby("trip_id").agg(
        total_dist_km=("distance_m", lambda x: x.sum() / 1000.0),
        mean_speed_ms=("speed_ms",   "mean"),
        actual_wh=    ("energy_wh",  "sum"),
        n_steps=      ("distance_m", "count"),
    ).reset_index()

    gain = df.groupby("trip_id")["elevation_m"].apply(elev_gain).reset_index()
    gain.columns = ["trip_id", "elevation_gain_m"]
    return agg.merge(gain, on="trip_id")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    model_dir = os.path.join(cfg.MODEL_DIR, "gps_best")

    with open(os.path.join(cfg.DATA_DIR, "osm_road_cache.pkl"), "rb") as f:
        osm_cache = pickle.load(f)

    print("Loading data...")
    train = train_best.add_road_features(
        pd.read_parquet(os.path.join(cfg.DATA_DIR, "train_trips.parquet")), osm_cache,
    )
    test = train_best.add_road_features(
        pd.read_parquet(os.path.join(cfg.DATA_DIR, "test_trips.parquet")), osm_cache,
    )

    for df in [train, test]:
        df["energy_whkm"] = np.where(
            df["distance_m"] > 0,
            df["energy_wh"] / (df["distance_m"] / 1000.0),
            np.nan,
        )

    gps_features = [f for f in cfg.GPS_FEATURES if f in test.columns and test[f].notna().any()]
    feature_cols  = gps_features + train_best.ROAD_FEATURE_COLS

    # Training-set mean Wh/km (constant baseline)
    train_trip = train.groupby("trip_id").agg(
        total_wh=  ("energy_wh",  "sum"),
        total_dist=("distance_m", "sum"),
    )
    train_trip["whkm"] = train_trip["total_wh"] / (train_trip["total_dist"] / 1000)
    train_mean_whkm = train_trip[train_trip["whkm"].between(0.1, 200)]["whkm"].mean()
    print(f"Training mean Wh/km: {train_mean_whkm:.2f}")

    # -----------------------------------------------------------------------
    # Multi-seed LSTM predictions
    # -----------------------------------------------------------------------
    print("\nPredicting with LSTM (multi-seed)...")
    scaler_path = os.path.join(model_dir, "lstm_scaler.joblib")
    seed_trip_dfs = {}

    for seed in train_best.SEEDS:
        mpath = os.path.join(model_dir, f"lstm_model_seed{seed}.pt")
        if not os.path.exists(mpath):
            print(f"  Seed {seed}: model not found, skipping")
            continue
        print(f"  Seed {seed}...", flush=True)
        preds = predict_lstm_from_path(mpath, scaler_path, test, feature_cols)
        tr    = build_trip_results(test, preds, "lstm_whkm")
        seed_trip_dfs[seed] = tr.set_index("trip_id")

    if not seed_trip_dfs:
        raise RuntimeError("No LSTM seed models found in " + model_dir)

    all_seeds = sorted(seed_trip_dfs.keys())
    first_df  = seed_trip_dfs[all_seeds[0]].sort_index()

    v_lstm = first_df[["actual_wh", "total_dist", "n_steps", "actual_whkm"]].copy().reset_index()
    for seed in all_seeds:
        v_lstm[f"pred_wh_s{seed}"] = v_lstm["trip_id"].map(seed_trip_dfs[seed]["pred_wh"])

    seed_cols = [f"pred_wh_s{s}" for s in all_seeds]
    v_lstm = v_lstm.dropna(subset=seed_cols).reset_index(drop=True)
    v_lstm["pred_wh"]   = v_lstm[seed_cols].mean(axis=1)
    v_lstm["lstm_whkm"] = v_lstm["pred_wh"] / (v_lstm["total_dist"] / 1000)
    v_lstm["mean_wh"]   = train_mean_whkm * (v_lstm["total_dist"] / 1000)

    w = v_lstm["n_steps"].values.astype(float)
    print(f"  Trips: {len(v_lstm):,}   Seeds used: {all_seeds}")

    # Per-seed metrics for std reporting
    per_seed_uw, per_seed_w = [], []
    for seed in all_seeds:
        pw = v_lstm[f"pred_wh_s{seed}"].values
        per_seed_uw.append(unweighted_metrics(v_lstm["actual_wh"].values, pw))
        per_seed_w.append(weighted_metrics(v_lstm["actual_wh"].values, pw, w))

    def seed_stat(metric_list, key):
        vals = [m[key] for m in metric_list]
        return float(np.mean(vals)), float(np.std(vals))

    # -----------------------------------------------------------------------
    # XGBoost predictions
    # -----------------------------------------------------------------------
    print("Predicting with XGBoost...")
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(model_dir, "xgb_model.json"))
    valid = test["energy_whkm"].notna() & np.isfinite(test["energy_whkm"])
    xgb_preds_raw = xgb_model.predict(test.loc[valid, feature_cols].fillna(0).values)

    tv = test[valid]
    ids = tv["trip_id"].values
    breaks = np.where(np.diff(ids) != 0)[0] + 1
    xgb_rows = []
    for s, e in zip(np.concatenate([[0], breaks]), np.concatenate([breaks, [len(ids)]])):
        if e - s > 0:
            c = tv.iloc[s:e]
            xgb_rows.append({
                "trip_id":    c["trip_id"].iloc[0],
                "xgb_whkm":  float(np.mean(xgb_preds_raw[s:e])),
                "actual_wh":  float(c["energy_wh"].sum()),
                "total_dist": float(c["distance_m"].sum()),
                "n_steps":    int(e - s),
            })
    v_xgb = pd.DataFrame(xgb_rows)
    v_xgb["actual_whkm"] = v_xgb["actual_wh"] / (v_xgb["total_dist"] / 1000)
    v_xgb["pred_wh"]     = v_xgb["xgb_whkm"]  * (v_xgb["total_dist"] / 1000)
    v_xgb = v_xgb[v_xgb["actual_whkm"].between(0.1, 200)].copy()

    # -----------------------------------------------------------------------
    # OLS baseline: trip-level LinearRegression on distance + elevation + speed
    # -----------------------------------------------------------------------
    print("Fitting OLS baseline...")
    train_trip_f = compute_ols_trip_features(train)
    test_trip_f  = compute_ols_trip_features(test)

    OLS_FEATS = ["total_dist_km", "elevation_gain_m", "mean_speed_ms"]

    # Filter to valid trips
    tr_ols = train_trip_f.dropna(subset=OLS_FEATS + ["actual_wh"])
    tr_ols = tr_ols[tr_ols["actual_wh"] > 0]
    ols = LinearRegression().fit(tr_ols[OLS_FEATS].values, tr_ols["actual_wh"].values)

    te_ols = test_trip_f.dropna(subset=OLS_FEATS + ["actual_wh"])
    te_ols = te_ols[te_ols["actual_wh"] > 0].copy()
    te_ols["pred_wh"]     = ols.predict(te_ols[OLS_FEATS].values)
    te_ols["actual_whkm"] = te_ols["actual_wh"] / te_ols["total_dist_km"]
    te_ols["pred_whkm"]   = te_ols["pred_wh"]   / te_ols["total_dist_km"]
    te_ols = te_ols[te_ols["actual_whkm"].between(0.1, 200)].copy()
    print(f"  OLS coefs: dist={ols.coef_[0]:.1f} Wh/km, elev={ols.coef_[1]:.3f} Wh/m, speed={ols.coef_[2]:.1f} Wh/(m/s)")

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    results = {}

    # LSTM (ensemble mean)
    w_lstm = v_lstm["n_steps"].values.astype(float)
    results["LSTM"] = {
        "unweighted_wh":   unweighted_metrics(v_lstm["actual_wh"].values, v_lstm["pred_wh"].values),
        "unweighted_whkm": unweighted_metrics(v_lstm["actual_whkm"].values, v_lstm["lstm_whkm"].values),
        "weighted_wh":     weighted_metrics(v_lstm["actual_wh"].values, v_lstm["pred_wh"].values, w_lstm),
        "weighted_whkm":   weighted_metrics(v_lstm["actual_whkm"].values, v_lstm["lstm_whkm"].values, w_lstm),
        "n_trips":         int(len(v_lstm)),
        "n_seeds":         len(all_seeds),
        "seed_std_wh_mae":  float(np.std([m["mae"]  for m in per_seed_uw])),
        "seed_std_wh_r2":   float(np.std([m["r2"]   for m in per_seed_uw])),
        "seed_std_mape":    float(np.std([m["mape"]  for m in per_seed_uw])),
    }

    # XGBoost
    w_xgb = v_xgb["n_steps"].values.astype(float)
    results["XGBoost"] = {
        "unweighted_wh":   unweighted_metrics(v_xgb["actual_wh"].values, v_xgb["pred_wh"].values),
        "unweighted_whkm": unweighted_metrics(v_xgb["actual_whkm"].values, v_xgb["xgb_whkm"].values),
        "weighted_wh":     weighted_metrics(v_xgb["actual_wh"].values, v_xgb["pred_wh"].values, w_xgb),
        "weighted_whkm":   weighted_metrics(v_xgb["actual_whkm"].values, v_xgb["xgb_whkm"].values, w_xgb),
        "n_trips":         int(len(v_xgb)),
    }

    # OLS baseline
    w_ols = te_ols["n_steps"].values.astype(float)
    results["OLS baseline"] = {
        "unweighted_wh":   unweighted_metrics(te_ols["actual_wh"].values, te_ols["pred_wh"].values),
        "unweighted_whkm": unweighted_metrics(te_ols["actual_whkm"].values, te_ols["pred_whkm"].values),
        "weighted_wh":     weighted_metrics(te_ols["actual_wh"].values, te_ols["pred_wh"].values, w_ols),
        "weighted_whkm":   weighted_metrics(te_ols["actual_whkm"].values, te_ols["pred_whkm"].values, w_ols),
        "n_trips":         int(len(te_ols)),
        "ols_coefs":       {"distance_km": float(ols.coef_[0]),
                            "elevation_gain_m": float(ols.coef_[1]),
                            "mean_speed_ms": float(ols.coef_[2]),
                            "intercept": float(ols.intercept_)},
    }

    # Constant mean baseline (on same trips as LSTM)
    mean_pred = v_lstm["mean_wh"].values
    results["Mean baseline"] = {
        "unweighted_wh":   unweighted_metrics(v_lstm["actual_wh"].values, mean_pred),
        "unweighted_whkm": unweighted_metrics(v_lstm["actual_whkm"].values,
                                               np.full(len(v_lstm), train_mean_whkm)),
        "weighted_wh":     weighted_metrics(v_lstm["actual_wh"].values, mean_pred, w_lstm),
        "weighted_whkm":   weighted_metrics(v_lstm["actual_whkm"].values,
                                             np.full(len(v_lstm), train_mean_whkm), w_lstm),
        "n_trips":         int(len(v_lstm)),
        "train_mean_whkm": float(train_mean_whkm),
    }

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TEST SET RESULTS — GPS-Only Models")
    print("=" * 100)

    hdr = f"{'Model':<18} {'Wh MAE':>8} {'Wh R²':>7} {'MAPE':>7} {'MdAPE':>7} {'Wh/km R²':>9}"
    sep = "-" * 65

    print(f"\n--- Unweighted (all trips equal) ---")
    print(hdr); print(sep)
    lstm_uw = results["LSTM"]["unweighted_wh"]
    lstm_uk = results["LSTM"]["unweighted_whkm"]
    std_mae = results["LSTM"]["seed_std_wh_mae"]
    std_r2  = results["LSTM"]["seed_std_wh_r2"]
    print(f"{'LSTM':<18} {lstm_uw['mae']:>8.2f} {lstm_uw['r2']:>7.3f} {lstm_uw['mape']:>6.1f}%"
          f" {lstm_uw['mdape']:>6.1f}% {lstm_uk['r2']:>9.3f}")
    print(f"{'  ± std (seeds)':<18} {std_mae:>8.2f} {std_r2:>7.3f}")

    for label in ["XGBoost", "OLS baseline", "Mean baseline"]:
        r  = results[label]
        u  = r["unweighted_wh"]
        uk = r["unweighted_whkm"]
        print(f"{label:<18} {u['mae']:>8.2f} {u['r2']:>7.3f} {u['mape']:>6.1f}%"
              f" {u['mdape']:>6.1f}% {uk['r2']:>9.3f}")

    print(f"\n--- Timestep-weighted (longer trips count more) ---")
    print(hdr); print(sep)
    lstm_ww = results["LSTM"]["weighted_wh"]
    lstm_wk = results["LSTM"]["weighted_whkm"]
    std_wmae = float(np.std([m["wmae"]  for m in per_seed_w]))
    std_wr2  = float(np.std([m["wr2"]   for m in per_seed_w]))
    print(f"{'LSTM':<18} {lstm_ww['wmae']:>8.2f} {lstm_ww['wr2']:>7.3f} {lstm_ww['wmape']:>6.1f}%"
          f" {lstm_ww['wmdape']:>6.1f}% {lstm_wk['wr2']:>9.3f}")
    print(f"{'  ± std (seeds)':<18} {std_wmae:>8.2f} {std_wr2:>7.3f}")

    for label in ["XGBoost", "OLS baseline", "Mean baseline"]:
        r  = results[label]
        wm = r["weighted_wh"]
        wk = r["weighted_whkm"]
        print(f"{label:<18} {wm['wmae']:>8.2f} {wm['wr2']:>7.3f} {wm['wmape']:>6.1f}%"
              f" {wm['wmdape']:>6.1f}% {wk['wr2']:>9.3f}")

    # Fleet-level sanity check
    total_actual  = float(v_lstm["actual_wh"].sum())
    total_lstm    = float(v_lstm["pred_wh"].sum())
    total_xgb     = float(v_xgb["pred_wh"].sum())
    print(f"\n--- Fleet-level total energy (test set, {len(v_lstm):,} trips) ---")
    print(f"  Actual:  {total_actual/1000:>8.1f} kWh")
    print(f"  LSTM:    {total_lstm/1000:>8.1f} kWh  (error {(total_lstm-total_actual)/total_actual*100:+.1f}%)")
    print(f"  XGBoost: {total_xgb/1000:>8.1f} kWh  (error {(total_xgb-total_actual)/total_actual*100:+.1f}%)")
    results["fleet_sanity"] = {
        "actual_kwh":        round(total_actual / 1000, 2),
        "lstm_kwh":          round(total_lstm   / 1000, 2),
        "xgb_kwh":           round(total_xgb    / 1000, 2),
        "lstm_error_pct":    round((total_lstm   - total_actual) / total_actual * 100, 2),
        "xgb_error_pct":     round((total_xgb    - total_actual) / total_actual * 100, 2),
    }

    # Clip sensitivity table (if 06d was run)
    sens_path = os.path.join(cfg.MODEL_DIR, "clip_sensitivity", "clip_sensitivity.json")
    if os.path.exists(sens_path):
        with open(sens_path) as f:
            sens = json.load(f)
        print(f"\n--- P90 Clip Sensitivity (XGBoost, validation set) ---")
        print(f"{'Level':<10} {'Clip (Wh/km)':>14} {'Val Wh MAE':>12} {'MAPE':>8} {'Wh/km R²':>10}")
        print("-" * 58)
        for row in sens:
            cv  = f"{row['clip_value_whkm']:.1f}" if row["clip_value_whkm"] else "none"
            mrk = " ←" if row["label"] == "p90" else ""
            print(f"{row['label']:<10} {cv:>14} {row['val_wh_mae']:>12.2f}"
                  f" {row['val_mape_pct']:>7.1f}% {row['val_whkm_r2']:>10.3f}{mrk}")
        results["clip_sensitivity"] = sens

    # -----------------------------------------------------------------------
    # Save metrics JSON
    # -----------------------------------------------------------------------
    def to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_native(v) for v in obj]
        return obj

    metrics_path = os.path.join(cfg.RESULTS_DIR, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(to_native(results), f, indent=2)
    print(f"\n  Saved → {metrics_path}")

    # -----------------------------------------------------------------------
    # Plot 1: Timestep-weighted error comparison (LSTM vs Mean)
    # -----------------------------------------------------------------------
    v_lstm["lstm_err_pct"] = (v_lstm["actual_wh"] - v_lstm["pred_wh"]) / v_lstm["actual_wh"] * 100
    v_lstm["mean_err_pct"] = (v_lstm["actual_wh"] - v_lstm["mean_wh"]) / v_lstm["actual_wh"] * 100
    w_norm = w_lstm / w_lstm.sum() * len(w_lstm)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    bins = np.linspace(-100, 100, 81)

    ax1 = axes[0]
    ax1.hist(v_lstm["lstm_err_pct"].clip(-100, 100), bins=bins,
             weights=w_norm, color="#3498db", edgecolor="white", alpha=0.85)
    ax1.axvline(0, color="red", linestyle="-", linewidth=1, alpha=0.5)
    ax1.axvline(np.average(v_lstm["lstm_err_pct"], weights=w_lstm),
                color="navy", linestyle="--", linewidth=1.5,
                label=f"Wtd mean: {np.average(v_lstm['lstm_err_pct'], weights=w_lstm):.1f}%")
    ax1.set_xlabel("Prediction Error %  (actual−pred)/actual", fontsize=11)
    ax1.set_ylabel("Weighted count (by n_steps)", fontsize=11)
    ax1.set_title(f"LSTM GPS Model  ({len(all_seeds)} seeds)\n(weighted by timestep count)",
                  fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_xlim(-100, 100)

    ax2 = axes[1]
    ax2.hist(v_lstm["mean_err_pct"].clip(-100, 100), bins=bins,
             weights=w_norm, color="#e74c3c", edgecolor="white", alpha=0.85)
    ax2.axvline(0, color="red", linestyle="-", linewidth=1, alpha=0.5)
    ax2.axvline(np.average(v_lstm["mean_err_pct"], weights=w_lstm),
                color="darkred", linestyle="--", linewidth=1.5,
                label=f"Wtd mean: {np.average(v_lstm['mean_err_pct'], weights=w_lstm):.1f}%")
    ax2.set_xlabel("Prediction Error %  (actual−pred)/actual", fontsize=11)
    ax2.set_ylabel("Weighted count (by n_steps)", fontsize=11)
    ax2.set_title(f"Mean Baseline ({train_mean_whkm:.1f} Wh/km)\n(weighted by timestep count)",
                  fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.set_xlim(-100, 100)

    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(0, ymax); ax2.set_ylim(0, ymax)

    ax3 = axes[2]
    ax3.axis("off")
    rl = results["LSTM"]; rm = results["Mean baseline"]
    lstm_abs = np.abs(v_lstm["lstm_err_pct"])
    mean_abs = np.abs(v_lstm["mean_err_pct"])
    mae_mean, mae_std = seed_stat(per_seed_uw, "mae")
    r2_mean,  r2_std  = seed_stat(per_seed_uw, "r2")
    text = (
        f"Timestep-Weighted Metrics\n"
        f"{'━'*42}\n"
        f"{'':>22} {'LSTM':>10} {'Mean':>10}\n"
        f"{'─'*42}\n"
        f"{'Wtd Wh MAE':>22} {rl['weighted_wh']['wmae']:>9.2f} {rm['weighted_wh']['wmae']:>9.2f}\n"
        f"{'Wtd Wh R²':>22} {rl['weighted_wh']['wr2']:>9.3f} {rm['weighted_wh']['wr2']:>9.3f}\n"
        f"{'Wtd MAPE':>22} {rl['weighted_wh']['wmape']:>8.1f}% {rm['weighted_wh']['wmape']:>8.1f}%\n"
        f"{'─'*42}\n"
        f"{'Wtd Wh/km R²':>22} {rl['weighted_whkm']['wr2']:>9.3f} {rm['weighted_whkm']['wr2']:>9.3f}\n"
        f"{'─'*42}\n"
        f"{'Trips':>22} {len(v_lstm):>9,}\n"
        f"{'Seeds':>22} {len(all_seeds):>9}\n"
        f"{'MAE (mean±std)':>22} {mae_mean:>7.2f}±{mae_std:.2f}\n"
        f"{'R² (mean±std)':>22} {r2_mean:>7.3f}±{r2_std:.3f}\n"
        f"\n"
        f"  LSTM reduces wtd MAE by\n"
        f"  {(1 - rl['weighted_wh']['wmae']/rm['weighted_wh']['wmae'])*100:.0f}% vs mean baseline\n"
        f"\n"
        f"  Wtd |err| < 20%:  LSTM {np.average(lstm_abs < 20, weights=w_lstm)*100:.0f}%"
        f"  Mean {np.average(mean_abs < 20, weights=w_lstm)*100:.0f}%\n"
        f"  Wtd |err| < 50%:  LSTM {np.average(lstm_abs < 50, weights=w_lstm)*100:.0f}%"
        f"  Mean {np.average(mean_abs < 50, weights=w_lstm)*100:.0f}%\n"
    )
    ax3.text(0.05, 0.95, text, transform=ax3.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#f7f7f7", alpha=0.9))

    plt.suptitle(
        "Trip Energy Prediction Error: LSTM vs Mean Baseline (Weighted by Timestep Count)",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.RESULTS_DIR, "lstm_vs_mean_error_weighted.png"),
                dpi=150, bbox_inches="tight")
    print(f"  Saved → results/lstm_vs_mean_error_weighted.png")

    # -----------------------------------------------------------------------
    # Plot 2: Predicted vs Actual Wh/km scatter
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(v_lstm["actual_whkm"], v_lstm["lstm_whkm"],
               alpha=0.15, s=8, c="#3498db", edgecolors="none")
    lims = [0, 120]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual Wh/km",    fontsize=12)
    ax.set_ylabel("Predicted Wh/km", fontsize=12)
    uk = results["LSTM"]["unweighted_whkm"]
    wk = results["LSTM"]["weighted_whkm"]
    ax.set_title(
        f"LSTM GPS Model: Trip-Level Wh/km\n"
        f"R²={uk['r2']:.3f} (unweighted)  |  R²={wk['wr2']:.3f} (timestep-weighted)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.RESULTS_DIR, "whkm_scatter.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved → results/whkm_scatter.png")

    # -----------------------------------------------------------------------
    # Plot 3: Feature importance (XGBoost)
    # -----------------------------------------------------------------------
    imp = dict(zip(feature_cols, xgb_model.feature_importances_))
    imp_sorted = sorted(imp.items(), key=lambda x: -x[1])
    label_map = {
        "speed_ms": "Speed (m/s)", "accel_ms2": "Acceleration", "jerk_ms3": "Jerk",
        "distance_m": "Step distance", "elevation_m": "Elevation", "slope_pct": "Road slope",
        "heading_sin": "Heading (sin)", "heading_cos": "Heading (cos)",
        "heading_change_rate": "Heading change rate", "curvature": "Curvature",
        "hour_sin": "Time of day (sin)", "hour_cos": "Time of day (cos)",
        "day_of_week": "Day of week", "dt_seconds": "Time interval",
        "speed_x_slope": "Speed × Slope", "kinetic_energy_proxy": "Kinetic energy",
        "road_type_ord": "Road class (ordinal)", "road_residential": "Residential road",
        "road_secondary": "Secondary road", "road_tertiary": "Tertiary road",
        "road_primary": "Primary road", "road_trunk": "Trunk road",
        "road_motorway": "Motorway",
    }

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    names  = [label_map.get(n, n) for n, _ in imp_sorted]
    scores = [s for _, s in imp_sorted]
    colors = ["#2ecc71" if "road" in imp_sorted[i][0] else "#3498db" for i in range(len(imp_sorted))]
    y_pos  = np.arange(len(names))
    ax.barh(y_pos, scores, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (gain)", fontsize=12)
    ax.set_title("GPS-Only XGBoost: Feature Importance", fontsize=14, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend([Patch(color="#3498db"), Patch(color="#2ecc71")],
              ["GPS-derived", "OSM road type"], loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.RESULTS_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved → results/feature_importance.png")

    print("\n" + "=" * 70)
    print("Evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
