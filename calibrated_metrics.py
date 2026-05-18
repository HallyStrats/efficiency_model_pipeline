"""
Compute calibrated trip-level metrics for the 3-seed LSTM ensemble and
emit the two paper figures:
  - figures/f_error_density.pdf      (KDE of trip-level error %, LSTM vs mean)
  - figures/f_efficiency_spread.pdf  (3-panel Wh/km spread: actual / mean / LSTM)
"""

import os, sys, pickle
os.environ["PYTORCH_MPS_DISABLE"] = "1"

import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from importlib import import_module

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
cfg = import_module("00_config")
train_best = import_module("06c_train_best")
ev = import_module("08_evaluate")

# Figures are written to research_paper/draft/figures/ when running from the
# author's working tree; if that directory does not exist (e.g. on a fresh
# clone of just this repo), fall back to ./figures/ next to this script.
_AUTHOR_FIG_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "research_paper", "draft", "figures"))
_LOCAL_FIG_DIR  = os.path.join(os.path.dirname(__file__), "figures")
FIG_DIR = _AUTHOR_FIG_DIR if os.path.isdir(_AUTHOR_FIG_DIR) else _LOCAL_FIG_DIR
os.makedirs(FIG_DIR, exist_ok=True)


def uw(actual, pred):
    return {
        "mae":   float(mean_absolute_error(actual, pred)),
        "rmse":  float(np.sqrt(mean_squared_error(actual, pred))),
        "r2":    float(r2_score(actual, pred)),
        "mape":  float(np.mean(np.abs((actual - pred) / np.clip(actual, 0.01, None))) * 100),
        "mdape": float(np.median(np.abs((actual - pred) / np.clip(actual, 0.01, None))) * 100),
        "bias":  float((pred - actual).mean()),
    }


def main():
    model_dir = os.path.join(cfg.MODEL_DIR, "gps_best")

    with open(os.path.join(cfg.DATA_DIR, "osm_road_cache.pkl"), "rb") as f:
        osm_cache = pickle.load(f)

    train = train_best.add_road_features(
        pd.read_parquet(os.path.join(cfg.DATA_DIR, "train_trips.parquet")), osm_cache)
    val   = train_best.add_road_features(
        pd.read_parquet(os.path.join(cfg.DATA_DIR, "val_trips.parquet")),   osm_cache)
    test  = train_best.add_road_features(
        pd.read_parquet(os.path.join(cfg.DATA_DIR, "test_trips.parquet")),  osm_cache)
    for df in [train, val, test]:
        df["energy_whkm"] = np.where(df["distance_m"] > 0, df["energy_wh"]/(df["distance_m"]/1000.0), np.nan)

    gps_features = [f for f in cfg.GPS_FEATURES if f in test.columns and test[f].notna().any()]
    feature_cols = gps_features + train_best.ROAD_FEATURE_COLS

    train_trip = train.groupby("trip_id").agg(total_wh=("energy_wh","sum"), total_dist=("distance_m","sum"))
    train_trip["whkm"] = train_trip["total_wh"]/(train_trip["total_dist"]/1000)
    train_mean_whkm = train_trip[train_trip["whkm"].between(0.1,200)]["whkm"].mean()
    val_trip = val.groupby("trip_id").agg(total_wh=("energy_wh","sum"), total_dist=("distance_m","sum"))
    val_mean_whkm = float(val_trip["total_wh"].sum() / (val_trip["total_dist"].sum()/1000))
    print(f"train_mean_whkm = {train_mean_whkm:.4f}   val_mean_whkm = {val_mean_whkm:.4f}   ratio = {val_mean_whkm/train_mean_whkm:.4f}")

    scaler_path = os.path.join(model_dir, "lstm_scaler.joblib")

    def predict_pack(df):
        seed_dfs = {}
        for seed in train_best.SEEDS:
            mp = os.path.join(model_dir, f"lstm_model_seed{seed}.pt")
            preds = ev.predict_lstm_from_path(mp, scaler_path, df, feature_cols)
            tr = ev.build_trip_results(df, preds, "lstm_whkm")
            seed_dfs[seed] = tr.set_index("trip_id")
        seeds = sorted(seed_dfs.keys())
        first = seed_dfs[seeds[0]].sort_index()
        v = first[["actual_wh","total_dist","n_steps","actual_whkm"]].copy().reset_index()
        for s in seeds:
            v[f"pred_wh_s{s}"] = v["trip_id"].map(seed_dfs[s]["pred_wh"])
        sc = [f"pred_wh_s{s}" for s in seeds]
        v = v.dropna(subset=sc).reset_index(drop=True)
        v["pred_wh"] = v[sc].mean(axis=1)
        v["mean_wh"] = val_mean_whkm * v["total_dist"]/1000
        return v, seeds

    print("Predicting on val ...");  v_val,  seeds = predict_pack(val)
    print("Predicting on test ..."); v_test, _     = predict_pack(test)

    scale = float(v_val["actual_wh"].sum() / v_val["pred_wh"].sum())
    print(f"Calibration scale = {scale:.4f}")

    cal_pred_wh = v_test["pred_wh"].values * scale
    actual_wh   = v_test["actual_wh"].values
    dist_km     = v_test["total_dist"].values / 1000.0
    actual_whkm = actual_wh / dist_km
    cal_pred_whkm = cal_pred_wh / dist_km
    mean_pred_whkm = np.full_like(actual_whkm, val_mean_whkm)

    cal       = uw(actual_wh,    cal_pred_wh)
    mean      = uw(actual_wh,    v_test["mean_wh"].values)
    cal_eff   = uw(actual_whkm,  cal_pred_whkm)
    mean_eff  = uw(actual_whkm,  mean_pred_whkm)

    print(f"[Wh    ] LSTM   MAE={cal['mae']:6.2f}  RMSE={cal['rmse']:6.2f}  R²={cal['r2']:.4f}  MAPE={cal['mape']:5.2f}  MdAPE={cal['mdape']:5.2f}  bias={cal['bias']:+.2f}")
    print(f"[Wh    ] Mean   MAE={mean['mae']:6.2f}  RMSE={mean['rmse']:6.2f}  R²={mean['r2']:.4f}  MAPE={mean['mape']:5.2f}  MdAPE={mean['mdape']:5.2f}  bias={mean['bias']:+.2f}")
    print(f"[Wh/km ] LSTM   MAE={cal_eff['mae']:6.2f}  RMSE={cal_eff['rmse']:6.2f}  R²={cal_eff['r2']:.4f}  MAPE={cal_eff['mape']:5.2f}  MdAPE={cal_eff['mdape']:5.2f}  bias={cal_eff['bias']:+.2f}")
    print(f"[Wh/km ] Mean   MAE={mean_eff['mae']:6.2f}  RMSE={mean_eff['rmse']:6.2f}  R²={mean_eff['r2']:.4f}  MAPE={mean_eff['mape']:5.2f}  MdAPE={mean_eff['mdape']:5.2f}  bias={mean_eff['bias']:+.2f}")

    # Per-seed std for Wh and Wh/km (each seed individually calibrated on val)
    seed_wh, seed_whkm = [], []
    for s in seeds:
        scale_s = float(v_val["actual_wh"].sum() / v_val[f"pred_wh_s{s}"].sum())
        cw   = v_test[f"pred_wh_s{s}"].values * scale_s
        cwkm = cw / dist_km
        seed_wh.append(uw(actual_wh, cw))
        seed_whkm.append(uw(actual_whkm, cwkm))
    def m_s(lst, key):
        v = [x[key] for x in lst]
        return float(np.mean(v)), float(np.std(v))
    print(f"Per-seed Wh   : MAE={m_s(seed_wh,'mae')[0]:.3f}±{m_s(seed_wh,'mae')[1]:.3f}  R²={m_s(seed_wh,'r2')[0]:.4f}±{m_s(seed_wh,'r2')[1]:.4f}")
    print(f"Per-seed Wh/km: MAE={m_s(seed_whkm,'mae')[0]:.3f}±{m_s(seed_whkm,'mae')[1]:.3f}  R²={m_s(seed_whkm,'r2')[0]:.4f}±{m_s(seed_whkm,'r2')[1]:.4f}")

    # Timestep-weighted trip-level Wh/km metrics (longer trips count more)
    w = v_test["n_steps"].values.astype(float)
    def wm(a, p, w):
        err = a - p; abs_err = np.abs(err); pct = np.abs(err / np.clip(a, 0.01, None))
        wmae = np.average(abs_err, weights=w)
        wrmse = np.sqrt(np.average(err**2, weights=w))
        wmape = np.average(pct, weights=w) * 100
        ss_res = np.average(err**2, weights=w)
        wm_ = np.average(a, weights=w)
        ss_tot = np.average((a - wm_)**2, weights=w)
        wr2 = 1 - ss_res/ss_tot
        sort_idx = np.argsort(pct); cum = np.cumsum(w[sort_idx])
        mid = np.searchsorted(cum, cum[-1]/2)
        wmdape = pct[sort_idx[min(mid, len(sort_idx)-1)]] * 100
        return {"mae": wmae, "rmse": wrmse, "r2": wr2, "mape": wmape, "mdape": wmdape}
    w_lstm_whkm = wm(actual_whkm,  cal_pred_whkm,            w)
    w_mean_whkm = wm(actual_whkm,  mean_pred_whkm,           w)
    w_lstm_wh   = wm(actual_wh,    cal_pred_wh,              w)
    w_mean_wh   = wm(actual_wh,    v_test["mean_wh"].values, w)
    print(f"[Wh wtd]    LSTM   MAE={w_lstm_wh['mae']:.2f}  RMSE={w_lstm_wh['rmse']:.2f}  R²={w_lstm_wh['r2']:.4f}  MAPE={w_lstm_wh['mape']:.2f}  MdAPE={w_lstm_wh['mdape']:.2f}")
    print(f"[Wh wtd]    Mean   MAE={w_mean_wh['mae']:.2f}  RMSE={w_mean_wh['rmse']:.2f}  R²={w_mean_wh['r2']:.4f}  MAPE={w_mean_wh['mape']:.2f}  MdAPE={w_mean_wh['mdape']:.2f}")
    print(f"[Wh/km wtd] LSTM   MAE={w_lstm_whkm['mae']:.2f}  RMSE={w_lstm_whkm['rmse']:.2f}  R²={w_lstm_whkm['r2']:.4f}  MAPE={w_lstm_whkm['mape']:.2f}  MdAPE={w_lstm_whkm['mdape']:.2f}")
    print(f"[Wh/km wtd] Mean   MAE={w_mean_whkm['mae']:.2f}  RMSE={w_mean_whkm['rmse']:.2f}  R²={w_mean_whkm['r2']:.4f}  MAPE={w_mean_whkm['mape']:.2f}  MdAPE={w_mean_whkm['mdape']:.2f}")
    # Per-seed weighted Wh and Wh/km
    seed_w_wh, seed_w_whkm = [], []
    for s in seeds:
        scale_s = float(v_val["actual_wh"].sum() / v_val[f"pred_wh_s{s}"].sum())
        cw   = v_test[f"pred_wh_s{s}"].values * scale_s
        cwkm = cw / dist_km
        seed_w_wh.append(  wm(actual_wh,   cw,   w))
        seed_w_whkm.append(wm(actual_whkm, cwkm, w))
    print(f"Per-seed Wh    wtd: MAE={m_s(seed_w_wh,'mae')[0]:.3f}±{m_s(seed_w_wh,'mae')[1]:.3f}  R²={m_s(seed_w_wh,'r2')[0]:.4f}±{m_s(seed_w_wh,'r2')[1]:.4f}")
    print(f"Per-seed Wh/km wtd: MAE={m_s(seed_w_whkm,'mae')[0]:.3f}±{m_s(seed_w_whkm,'mae')[1]:.3f}  R²={m_s(seed_w_whkm,'r2')[0]:.4f}±{m_s(seed_w_whkm,'r2')[1]:.4f}")
    # Hand off to plotting code
    r2_lstm_whkm = w_lstm_whkm["r2"]
    r2_mean_whkm = w_mean_whkm["r2"]
    mae_lstm_wh_used = w_lstm_wh["mae"]
    mae_mean_wh_used = w_mean_wh["mae"]

    # Per-timestep Wh/km R²: load val + test step-level predictions for the
    # 3-seed ensemble and compute step-level R² against per-step Wh/km labels.
    print("Computing per-timestep Wh/km R² (loading per-step preds) ...")
    per_step_actual, per_step_pred = [], []
    valid = test["energy_whkm"].notna() & np.isfinite(test["energy_whkm"])
    tv = test[valid].reset_index(drop=True)
    seed_step_preds = {}
    for s in seeds:
        mp = os.path.join(model_dir, f"lstm_model_seed{s}.pt")
        seed_step_preds[s] = ev.predict_lstm_from_path(mp, scaler_path, test, feature_cols)
    ids = tv["trip_id"].values
    breaks = np.where(np.diff(ids) != 0)[0] + 1
    starts = np.concatenate([[0], breaks])
    ends   = np.concatenate([breaks, [len(ids)]])
    step_actual_all = []; step_pred_ens = []
    for idx, (st, en) in enumerate(zip(starts, ends)):
        if idx >= len(seed_step_preds[seeds[0]]) or en - st <= 0:
            continue
        chunk_actual = tv["energy_whkm"].values[st:en]
        chunk_preds = np.mean(
            np.stack([seed_step_preds[s][idx][:en-st] for s in seeds], axis=0), axis=0)
        step_actual_all.append(chunk_actual)
        step_pred_ens.append(chunk_preds)
    step_actual = np.concatenate(step_actual_all)
    step_pred   = np.concatenate(step_pred_ens) * scale  # apply calibration
    # Apply same data filter as evaluation: clip extreme labels
    keep = np.isfinite(step_actual) & np.isfinite(step_pred)
    step_actual = step_actual[keep]; step_pred = step_pred[keep]
    r2_step_uncal = float(r2_score(step_actual, step_pred / scale))
    r2_step_cal   = float(r2_score(step_actual, step_pred))
    # Mean-baseline at step level
    r2_step_mean  = float(r2_score(step_actual, np.full_like(step_actual, train_mean_whkm)))
    print(f"[Wh/km, per-timestep] LSTM (cal) R²={r2_step_cal:.4f}   LSTM (uncal) R²={r2_step_uncal:.4f}   Mean R²={r2_step_mean:.4f}   n_steps={len(step_actual):,}")

    # Headline reported in paper: timestep-weighted Wh/km R² (assigned above).

    # ------------------------------------------------------------------
    # Figure 1: KDE of trip-level error % (calibrated LSTM vs mean)
    # ------------------------------------------------------------------
    lstm_err = ((actual_wh - cal_pred_wh) / actual_wh * 100).clip(-100, 100)
    mean_err = ((actual_wh - v_test["mean_wh"].values) / actual_wh * 100).clip(-100, 100)

    grid = np.linspace(-100, 100, 800)
    # Timestep-weighted KDE (longer trips count more)
    kde_l = gaussian_kde(lstm_err, weights=w, bw_method=0.25)(grid)
    kde_m = gaussian_kde(mean_err, weights=w, bw_method=0.25)(grid)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.fill_between(grid, kde_l, color="#1f77b4", alpha=0.25)
    ax.fill_between(grid, kde_m, color="#d62728", alpha=0.20)
    ax.plot(grid, kde_l, color="#1f77b4", linewidth=2.0,
            label=f"LSTM (MAE = {mae_lstm_wh_used:.2f} Wh)")
    ax.plot(grid, kde_m, color="#d62728", linewidth=2.0,
            label=f"Mean baseline (MAE = {mae_mean_wh_used:.2f} Wh)")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("Trip-level prediction error (%)  $(actual-pred)/actual$")
    ax.set_ylabel("Density")
    ax.set_xlim(-100, 100); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.25, linewidth=0.5)
    plt.tight_layout()
    out1 = os.path.join(FIG_DIR, "f_error_density.pdf")
    plt.savefig(out1, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out1}")

    # ------------------------------------------------------------------
    # Figure 2: 3-panel Wh/km spread (actual / mean predictor / LSTM)
    # ------------------------------------------------------------------
    lo, hi = 0, 120  # Wh/km display range
    bins = np.linspace(lo, hi, 61)
    centres = 0.5*(bins[:-1] + bins[1:])

    h_actual, _ = np.histogram(np.clip(actual_whkm,    lo, hi), bins=bins, weights=w)
    h_mean,   _ = np.histogram(np.clip(mean_pred_whkm, lo, hi), bins=bins, weights=w)
    h_lstm,   _ = np.histogram(np.clip(cal_pred_whkm,  lo, hi), bins=bins, weights=w)
    ymax_ac = max(h_actual.max(), h_lstm.max()) * 1.10
    ymax_b  = h_mean.max() * 1.10

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

    ax = axes[0]
    ax.bar(centres, h_actual, width=(bins[1]-bins[0])*0.95,
           color="#7f7f7f", edgecolor="white", alpha=0.85, label="Actual")
    ax.set_title("(a) Actual trip efficiency", fontsize=12)
    ax.set_xlabel("Trip efficiency (Wh/km)")
    ax.set_ylabel("Timesteps (count)")
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.5)

    ax = axes[1]
    ax.bar(centres, h_mean, width=(bins[1]-bins[0])*0.95,
           color="#d62728", edgecolor="white", alpha=0.85,
           label=f"Mean baseline ($R^2 = {r2_mean_whkm:+.3f}$)")
    ax.set_title("(b) Mean baseline predictions", fontsize=12)
    ax.set_xlabel("Predicted efficiency (Wh/km)")
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.5)

    ax = axes[2]
    ax.bar(centres, h_lstm, width=(bins[1]-bins[0])*0.95,
           color="#1f77b4", edgecolor="white", alpha=0.85,
           label=f"LSTM ($R^2 = {r2_lstm_whkm:+.3f}$)")
    ax.set_title("(c) LSTM predictions", fontsize=12)
    ax.set_xlabel("Predicted efficiency (Wh/km)")
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.5)

    for ax in axes:
        ax.set_xlim(lo, hi)
    axes[0].set_ylim(0, ymax_ac)
    axes[2].set_ylim(0, ymax_ac)
    axes[1].set_ylim(0, ymax_b)
    axes[1].set_ylabel("Timesteps (count)")
    axes[2].set_ylabel("Timesteps (count)")
    plt.tight_layout()
    out2 = os.path.join(FIG_DIR, "f_efficiency_spread.pdf")
    plt.savefig(out2, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out2}")

    # ------------------------------------------------------------------
    # Figure 3: MAE (Wh) vs trip distance
    # ------------------------------------------------------------------
    dist = dist_km
    safe_km = np.clip(dist_km, 1e-3, None)
    lstm_pred_whkm_trip = cal_pred_wh / safe_km
    mean_pred_whkm_trip = v_test["mean_wh"].values / safe_km
    actual_whkm_trip = actual_wh / safe_km
    lstm_abs = np.abs(actual_whkm_trip - lstm_pred_whkm_trip)
    mean_abs = np.abs(actual_whkm_trip - mean_pred_whkm_trip)

    bin_edges = np.array([0.0, 0.5, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0, 50.0])
    bin_lbls  = ["<0.5", "0.5–1", "1–1.5", "1.5–2.5", "2.5–4", "4–6", "6–10", ">10"]
    bidx = np.digitize(dist, bin_edges) - 1
    bidx = np.clip(bidx, 0, len(bin_lbls)-1)
    centres = np.arange(len(bin_lbls))

    def bin_mae(values, weights, idx):
        mae, n = [], []
        for b in range(len(bin_lbls)):
            sel = idx == b
            if sel.sum() == 0:
                mae.append(np.nan); n.append(0); continue
            mae.append(float(np.average(values[sel], weights=weights[sel])))
            n.append(int(sel.sum()))
        return np.array(mae), np.array(n)

    l_mae, n_lstm = bin_mae(lstm_abs, w, bidx)
    m_mae, _      = bin_mae(mean_abs, w, bidx)

    def bin_wstd(values, weights, idx):
        out = []
        for b in range(len(bin_lbls)):
            sel = idx == b
            if sel.sum() == 0:
                out.append(np.nan); continue
            v = values[sel]; ws = weights[sel]
            mu = np.average(v, weights=ws)
            out.append(float(np.sqrt(np.average((v - mu)**2, weights=ws))))
        return np.array(out)

    actual_std = bin_wstd(actual_whkm_trip, w, bidx)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(centres, l_mae, "-o", color="#1f77b4", linewidth=2.2, label="LSTM")
    ax.plot(centres, m_mae, "-s", color="#d62728", linewidth=2.2, label="Mean baseline")
    ax.set_ylabel("Trip MAE (Wh/km)")
    ax.set_xlabel("Trip distance (km)")
    ax.set_xticks(centres); ax.set_xticklabels(bin_lbls)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.25, linewidth=0.5)
    for c, n_ in zip(centres, n_lstm):
        ax.text(c, ax.get_ylim()[1]*0.97, f"n={n_:,}",
                ha="center", va="top", fontsize=8, color="0.35")

    plt.tight_layout()
    out3 = os.path.join(FIG_DIR, "f_error_vs_distance.pdf")
    plt.savefig(out3, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out3}")


if __name__ == "__main__":
    main()
