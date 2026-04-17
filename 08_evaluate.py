"""
08_evaluate.py

Evaluates all GPS-only models on the held-out test set.

Models evaluated:
  - LSTM (best model): per-timestep Wh/km → trip aggregation
  - XGBoost (tuned): per-timestep Wh/km → trip aggregation
  - Mean baseline: training-set mean Wh/km applied to all trips

Metrics:
  - Unweighted: MAE, RMSE, R², MAPE, MdAPE (all trips equal)
  - Timestep-weighted: weighted by n_steps per trip (longer trips count more,
    justified because short trips have higher target noise from rc sensor)

Output:
  results/metrics_summary.json
  results/lstm_vs_mean_error_weighted.png
  results/feature_importance.png
  results/whkm_scatter.png
"""

import json
import os
import pickle

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from importlib import import_module
cfg = import_module("00_config")
train_best = import_module("06c_train_best")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def weighted_metrics(actual, pred, weights):
    """Compute timestep-weighted regression metrics."""
    err = actual - pred
    abs_err = np.abs(err)
    pct_err = np.abs(err / np.clip(actual, 0.01, None))

    wmae = np.average(abs_err, weights=weights)
    wrmse = np.sqrt(np.average(err**2, weights=weights))
    wmape = np.average(pct_err, weights=weights) * 100

    ss_res = np.average(err**2, weights=weights)
    w_mean = np.average(actual, weights=weights)
    ss_tot = np.average((actual - w_mean)**2, weights=weights)
    wr2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Weighted median APE
    sort_idx = np.argsort(pct_err)
    cum_w = np.cumsum(weights[sort_idx])
    wmedian_idx = np.searchsorted(cum_w, cum_w[-1] / 2)
    wmdape = pct_err[sort_idx[min(wmedian_idx, len(sort_idx)-1)]] * 100

    bias = np.average(err, weights=weights)
    return {"wmae": wmae, "wrmse": wrmse, "wr2": wr2, "wmape": wmape, "wmdape": wmdape, "bias": bias}


def unweighted_metrics(actual, pred):
    """Compute standard regression metrics."""
    return {
        "mae": mean_absolute_error(actual, pred),
        "rmse": np.sqrt(mean_squared_error(actual, pred)),
        "r2": r2_score(actual, pred),
        "mape": np.mean(np.abs((actual - pred) / np.clip(actual, 0.01, None))) * 100,
        "mdape": np.median(np.abs((actual - pred) / np.clip(actual, 0.01, None))) * 100,
        "bias": (pred - actual).mean(),
    }


def predict_lstm(model_dir, test_df, feature_cols):
    """Load LSTM and predict per-timestep Wh/km on test data."""
    scaler = joblib.load(os.path.join(model_dir, "lstm_scaler.joblib"))
    test_sc = test_df.copy()
    test_sc[feature_cols] = scaler.transform(test_df[feature_cols].fillna(0))

    test_ds = train_best.TripDataset(test_sc, feature_cols, "energy_whkm")

    model = train_best.TripLSTM(input_size=len(feature_cols))
    model.load_state_dict(torch.load(
        os.path.join(model_dir, "lstm_model.pt"), weights_only=True,
    ))
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
    """Aggregate per-timestep predictions to trip level."""
    valid = test_df["energy_whkm"].notna() & np.isfinite(test_df["energy_whkm"])
    tv = test_df[valid]
    ids = tv["trip_id"].values
    breaks = np.where(np.diff(ids) != 0)[0] + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(ids)]])

    rows = []
    for idx, (s, e) in enumerate(zip(starts, ends)):
        if idx < len(all_preds) and e - s > 0:
            c = tv.iloc[s:e]
            rows.append({
                "trip_id": c["trip_id"].iloc[0],
                pred_col: np.mean(all_preds[idx][:e-s]),
                "actual_wh": c["energy_wh"].sum(),
                "total_dist": c["distance_m"].sum(),
                "n_steps": e - s,
            })

    tr = pd.DataFrame(rows)
    tr["actual_whkm"] = tr["actual_wh"] / (tr["total_dist"] / 1000.0)
    tr["pred_wh"] = tr[pred_col] * (tr["total_dist"] / 1000.0)
    return tr[tr["actual_whkm"].between(0.1, 200) & np.isfinite(tr["actual_whkm"])].copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    model_dir = os.path.join(cfg.MODEL_DIR, "gps_best")

    # Load OSM cache
    with open(os.path.join(cfg.DATA_DIR, "osm_road_cache.pkl"), "rb") as f:
        osm_cache = pickle.load(f)

    # Load data
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

    # Feature list
    gps_features = [f for f in cfg.GPS_FEATURES if f in test.columns and test[f].notna().any()]
    feature_cols = gps_features + train_best.ROAD_FEATURE_COLS

    # Training mean Wh/km (baseline)
    train_trip = train.groupby("trip_id").agg(
        total_wh=("energy_wh", "sum"), total_dist=("distance_m", "sum"),
    )
    train_trip["whkm"] = train_trip["total_wh"] / (train_trip["total_dist"] / 1000)
    train_mean_whkm = train_trip[train_trip["whkm"].between(0.1, 200)]["whkm"].mean()
    print(f"Training mean Wh/km: {train_mean_whkm:.2f}")

    # --- LSTM predictions ---
    print("\nPredicting with LSTM...")
    lstm_preds = predict_lstm(model_dir, test, feature_cols)
    v_lstm = build_trip_results(test, lstm_preds, "lstm_whkm")
    v_lstm["mean_wh"] = train_mean_whkm * (v_lstm["total_dist"] / 1000.0)

    # --- XGBoost predictions ---
    print("Predicting with XGBoost...")
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(model_dir, "xgb_model.json"))
    valid = test["energy_whkm"].notna() & np.isfinite(test["energy_whkm"])
    xgb_preds_raw = xgb_model.predict(test.loc[valid, feature_cols].fillna(0).values)

    # Build XGB trip results (same structure)
    tv = test[valid]
    ids = tv["trip_id"].values
    breaks = np.where(np.diff(ids) != 0)[0] + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(ids)]])
    xgb_rows = []
    offset = 0
    for s, e in zip(starts, ends):
        if e - s > 0:
            c = tv.iloc[s:e]
            xgb_rows.append({
                "trip_id": c["trip_id"].iloc[0],
                "xgb_whkm": np.mean(xgb_preds_raw[s:e]),
                "actual_wh": c["energy_wh"].sum(),
                "total_dist": c["distance_m"].sum(),
                "n_steps": e - s,
            })
    v_xgb = pd.DataFrame(xgb_rows)
    v_xgb["actual_whkm"] = v_xgb["actual_wh"] / (v_xgb["total_dist"] / 1000)
    v_xgb["pred_wh"] = v_xgb["xgb_whkm"] * (v_xgb["total_dist"] / 1000)
    v_xgb = v_xgb[v_xgb["actual_whkm"].between(0.1, 200)].copy()

    # --- Compute metrics ---
    v = v_lstm  # Use LSTM trip set as reference
    w = v["n_steps"].values.astype(float)

    results = {}
    for label, actual_wh, pred_wh, actual_whkm, pred_whkm in [
        ("LSTM", v["actual_wh"].values, v["pred_wh"].values,
         v["actual_whkm"].values, v["lstm_whkm"].values),
        ("XGBoost", v_xgb["actual_wh"].values, v_xgb["pred_wh"].values,
         v_xgb["actual_whkm"].values, v_xgb["xgb_whkm"].values),
        ("Mean baseline", v["actual_wh"].values, v["mean_wh"].values,
         v["actual_whkm"].values, np.full(len(v), train_mean_whkm)),
    ]:
        w_use = w if label != "XGBoost" else v_xgb["n_steps"].values.astype(float)
        r = {
            "unweighted_wh": unweighted_metrics(actual_wh, pred_wh),
            "unweighted_whkm": unweighted_metrics(actual_whkm, pred_whkm),
            "weighted_wh": weighted_metrics(actual_wh, pred_wh, w_use),
            "weighted_whkm": weighted_metrics(actual_whkm, pred_whkm, w_use),
            "n_trips": len(actual_wh),
        }
        results[label] = r

    # Print summary
    print("\n" + "=" * 90)
    print("TEST SET RESULTS (GPS-Only Models)")
    print("=" * 90)

    print(f"\n--- Unweighted (all trips equal) ---")
    print(f"{'Model':<20} {'Wh MAE':>8} {'Wh R²':>7} {'MAPE':>7} {'MdAPE':>7} {'Wh/km R²':>9}")
    print("-" * 65)
    for label in ["LSTM", "XGBoost", "Mean baseline"]:
        r = results[label]
        u = r["unweighted_wh"]
        uk = r["unweighted_whkm"]
        print(f"{label:<20} {u['mae']:>8.2f} {u['r2']:>7.3f} {u['mape']:>6.1f}% {u['mdape']:>6.1f}% {uk['r2']:>9.3f}")

    print(f"\n--- Timestep-weighted (longer trips count more) ---")
    print(f"{'Model':<20} {'Wh MAE':>8} {'Wh R²':>7} {'MAPE':>7} {'MdAPE':>7} {'Wh/km R²':>9}")
    print("-" * 65)
    for label in ["LSTM", "XGBoost", "Mean baseline"]:
        r = results[label]
        w_r = r["weighted_wh"]
        wk = r["weighted_whkm"]
        print(f"{label:<20} {w_r['wmae']:>8.2f} {w_r['wr2']:>7.3f} {w_r['wmape']:>6.1f}% {w_r['wmdape']:>6.1f}% {wk['wr2']:>9.3f}")

    # Save metrics
    # Convert numpy types for JSON serialisation
    def to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        return obj

    metrics_path = os.path.join(cfg.RESULTS_DIR, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(to_native(results), f, indent=2)
    print(f"\n  Saved → {metrics_path}")

    # --- Plot 1: Weighted error comparison (LSTM vs Mean) ---
    v["lstm_err_pct"] = (v["actual_wh"].values - v["pred_wh"].values) / v["actual_wh"].values * 100
    v["mean_err_pct"] = (v["actual_wh"].values - v["mean_wh"].values) / v["actual_wh"].values * 100
    w_norm = w / w.sum() * len(w)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    bins = np.linspace(-100, 100, 81)

    ax1 = axes[0]
    ax1.hist(v["lstm_err_pct"].clip(-100, 100), bins=bins, weights=w_norm, color="#3498db", edgecolor="white", alpha=0.85)
    ax1.axvline(0, color="red", linestyle="-", linewidth=1, alpha=0.5)
    ax1.axvline(np.average(v["lstm_err_pct"], weights=w), color="navy", linestyle="--", linewidth=1.5,
                label=f"Wtd mean: {np.average(v['lstm_err_pct'], weights=w):.1f}%")
    ax1.set_xlabel("Prediction Error %  (actual-pred)/actual", fontsize=11)
    ax1.set_ylabel("Weighted count (by n_steps)", fontsize=11)
    ax1.set_title("LSTM GPS Model\n(weighted by timestep count)", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_xlim(-100, 100)

    ax2 = axes[1]
    ax2.hist(v["mean_err_pct"].clip(-100, 100), bins=bins, weights=w_norm, color="#e74c3c", edgecolor="white", alpha=0.85)
    ax2.axvline(0, color="red", linestyle="-", linewidth=1, alpha=0.5)
    ax2.axvline(np.average(v["mean_err_pct"], weights=w), color="darkred", linestyle="--", linewidth=1.5,
                label=f"Wtd mean: {np.average(v['mean_err_pct'], weights=w):.1f}%")
    ax2.set_xlabel("Prediction Error %  (actual-pred)/actual", fontsize=11)
    ax2.set_ylabel("Weighted count (by n_steps)", fontsize=11)
    ax2.set_title(f"Mean Baseline ({train_mean_whkm:.1f} Wh/km)\n(weighted by timestep count)", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.set_xlim(-100, 100)

    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(0, ymax)
    ax2.set_ylim(0, ymax)

    ax3 = axes[2]
    ax3.axis("off")
    rl = results["LSTM"]
    rm = results["Mean baseline"]
    lstm_abs = np.abs(v["lstm_err_pct"])
    mean_abs = np.abs(v["mean_err_pct"])
    text = (
        f"Timestep-Weighted Metrics\n"
        f"{'━'*40}\n"
        f"{'':>20} {'LSTM':>10} {'Mean':>10}\n"
        f"{'─'*40}\n"
        f"{'Wtd Wh MAE':>20} {rl['weighted_wh']['wmae']:>9.2f} {rm['weighted_wh']['wmae']:>9.2f}\n"
        f"{'Wtd Wh R²':>20} {rl['weighted_wh']['wr2']:>9.3f} {rm['weighted_wh']['wr2']:>9.3f}\n"
        f"{'Wtd MAPE':>20} {rl['weighted_wh']['wmape']:>8.1f}% {rm['weighted_wh']['wmape']:>8.1f}%\n"
        f"{'Wtd MdAPE':>20} {rl['weighted_wh']['wmdape']:>8.1f}% {rm['weighted_wh']['wmdape']:>8.1f}%\n"
        f"{'─'*40}\n"
        f"{'Wtd Wh/km R²':>20} {rl['weighted_whkm']['wr2']:>9.3f} {rm['weighted_whkm']['wr2']:>9.3f}\n"
        f"{'─'*40}\n"
        f"{'Trips':>20} {len(v):>9,}\n"
        f"\n"
        f"  LSTM reduces wtd MAE by\n"
        f"  {(1 - rl['weighted_wh']['wmae']/rm['weighted_wh']['wmae'])*100:.0f}% vs mean baseline\n"
        f"\n"
        f"  Wtd |err| < 20%:\n"
        f"    LSTM {np.average(lstm_abs < 20, weights=w)*100:.0f}%"
        f"  Mean {np.average(mean_abs < 20, weights=w)*100:.0f}%\n"
        f"  Wtd |err| < 50%:\n"
        f"    LSTM {np.average(lstm_abs < 50, weights=w)*100:.0f}%"
        f"  Mean {np.average(mean_abs < 50, weights=w)*100:.0f}%\n"
    )
    ax3.text(0.05, 0.95, text, transform=ax3.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#f7f7f7", alpha=0.9))

    plt.suptitle("Trip Energy Prediction Error: LSTM vs Mean Baseline (Weighted by Timestep Count)",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.RESULTS_DIR, "lstm_vs_mean_error_weighted.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved → results/lstm_vs_mean_error_weighted.png")

    # --- Plot 2: Predicted vs Actual Wh/km scatter ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(v["actual_whkm"], v["lstm_whkm"], alpha=0.15, s=8, c="#3498db", edgecolors="none")
    lims = [0, 120]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Wh/km", fontsize=12)
    ax.set_ylabel("Predicted Wh/km", fontsize=12)
    uk = results["LSTM"]["unweighted_whkm"]
    wk = results["LSTM"]["weighted_whkm"]
    ax.set_title(f"LSTM GPS Model: Trip-Level Wh/km\nR²={uk['r2']:.3f} (unweighted)  |  R²={wk['wr2']:.3f} (timestep-weighted)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.RESULTS_DIR, "whkm_scatter.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved → results/whkm_scatter.png")

    # --- Plot 3: Feature importance (XGBoost) ---
    imp = dict(zip(feature_cols, xgb_model.feature_importances_))
    imp_sorted = sorted(imp.items(), key=lambda x: -x[1])

    label_map = {
        "speed_ms": "Speed (m/s)", "accel_ms2": "Acceleration", "jerk_ms3": "Jerk",
        "distance_m": "Step distance", "elevation_m": "Elevation", "slope_pct": "Road slope",
        "heading_sin": "Heading (sin)", "heading_cos": "Heading (cos)",
        "heading_change_rate": "Heading change rate", "curvature": "Curvature",
        "hour_sin": "Time of day (sin)", "hour_cos": "Time of day (cos)",
        "day_of_week": "Day of week", "dt_seconds": "Time interval",
        "speed_x_slope": "Speed x Slope", "kinetic_energy_proxy": "Kinetic energy",
        "road_type_ord": "Road class (ordinal)", "road_residential": "Residential road",
        "road_secondary": "Secondary road", "road_tertiary": "Tertiary road",
        "road_primary": "Primary road", "road_trunk": "Trunk road", "road_motorway": "Motorway",
    }

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    names = [label_map.get(n, n) for n, _ in imp_sorted]
    scores = [s for _, s in imp_sorted]
    colors = ["#2ecc71" if "road" in imp_sorted[i][0] else "#3498db" for i in range(len(imp_sorted))]

    y_pos = np.arange(len(names))
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
