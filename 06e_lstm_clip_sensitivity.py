"""
06e_lstm_clip_sensitivity.py

Compares the headline GPS LSTM (single seed = 42) trained at four target-clip
levels: P90 (current baseline), P95, P99, and no clip.  All other settings —
features, architecture, optimizer, scheduler, scaler, train/val split — are
held identical to 06c so the comparison isolates the clip effect.

For each clip level we report on the held-out test set:
  - Val/test MAE, RMSE, R², MAPE, bias on Wh/km and Wh
  - Fleet-level error %
  - Validation-set calibration factor and post-calibration metrics

Output:
  models/lstm_clip_sensitivity/lstm_seed42_{label}.pt
  results_new/lstm_clip_sensitivity.json
"""

import os, json, pickle
os.environ["PYTORCH_MPS_DISABLE"] = "1"

import joblib, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from importlib import import_module

cfg        = import_module("00_config")
train_best = import_module("06c_train_best")

OUT_DIR = os.path.join(cfg.MODEL_DIR, "lstm_clip_sensitivity")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
CLIP_LEVELS = [
    ("p90",     0.90),
    ("p95",     0.95),
    ("p99",     0.99),
    ("no_clip", None),
]


# ---------------------------------------------------------------------------
def train_lstm(train_df, val_df, feature_cols, target_col, model_path, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    train_ds = train_best.TripDataset(train_df, feature_cols, target_col)
    val_ds   = train_best.TripDataset(val_df,   feature_cols, target_col)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              collate_fn=train_best.collate_trips)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False,
                              collate_fn=train_best.collate_trips)
    device = torch.device("cpu")
    model = train_best.TripLSTM(input_size=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.HuberLoss(delta=1.0)
    best_val = float("inf"); patience = 0
    for epoch in range(100):
        model.train(); train_loss, n_b = 0, 0
        for seqs, targets, lengths in train_loader:
            preds = model(seqs.to(device), lengths)
            mask = torch.arange(targets.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
            loss = criterion(preds[mask], targets.to(device)[mask])
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item(); n_b += 1
        model.eval(); val_loss, n_v = 0, 0
        with torch.no_grad():
            for seqs, targets, lengths in val_loader:
                preds = model(seqs.to(device), lengths)
                mask = torch.arange(targets.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
                val_loss += criterion(preds[mask], targets.to(device)[mask]).item()
                n_v += 1
        avg_train = train_loss / max(n_b, 1); avg_val = val_loss / max(n_v, 1)
        scheduler.step(avg_val)
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:3d}: train={avg_train:.4f}  val={avg_val:.4f}", flush=True)
        if avg_val < best_val:
            best_val = avg_val; patience = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience += 1
            if patience >= 15:
                print(f"    Early stopping at epoch {epoch+1}", flush=True); break
    print(f"    Best val loss: {best_val:.4f}", flush=True)
    return model_path


def predict(model_path, scaler, df, feature_cols):
    sc = df.copy(); sc[feature_cols] = scaler.transform(df[feature_cols].fillna(0))
    ds = train_best.TripDataset(sc, feature_cols, "energy_whkm")
    model = train_best.TripLSTM(input_size=len(feature_cols))
    model.load_state_dict(torch.load(model_path, weights_only=True)); model.eval()
    out = []
    with torch.no_grad():
        for i in range(len(ds)):
            seq, _ = ds[i]; length = torch.tensor([len(ds.sequences[i])])
            out.append(model(seq.unsqueeze(0), length).numpy().flatten()[:len(ds.sequences[i])])
    return out


def aggregate_to_trip(df, all_preds):
    valid = df["energy_whkm"].notna() & np.isfinite(df["energy_whkm"])
    tv = df[valid]; ids = tv["trip_id"].values
    breaks = np.where(np.diff(ids) != 0)[0] + 1
    starts = np.concatenate([[0], breaks]); ends = np.concatenate([breaks, [len(ids)]])
    rows = []
    for idx, (s, e) in enumerate(zip(starts, ends)):
        if idx < len(all_preds) and e - s > 0:
            c = tv.iloc[s:e]
            rows.append({"trip_id": c["trip_id"].iloc[0],
                         "lstm_whkm":  float(np.mean(all_preds[idx][:e - s])),
                         "actual_wh":  float(c["energy_wh"].sum()),
                         "total_dist": float(c["distance_m"].sum()),
                         "n_steps":    int(e - s)})
    tr = pd.DataFrame(rows)
    tr["actual_whkm"] = tr["actual_wh"] / (tr["total_dist"]/1000.0)
    tr["pred_wh"] = tr["lstm_whkm"] * (tr["total_dist"]/1000.0)
    return tr[tr["actual_whkm"].between(0.1, 200) & np.isfinite(tr["actual_whkm"])].copy()


def metrics_w(actual, pred, weights):
    err = pred - actual
    w = weights / weights.sum()
    a_mean = np.average(actual, weights=w)
    ss_res = np.average(err**2, weights=w)
    ss_tot = np.average((actual - a_mean)**2, weights=w)
    return {
        "mae":  float(np.average(np.abs(err), weights=w)),
        "rmse": float(np.sqrt(np.average(err**2, weights=w))),
        "r2":   float(1 - ss_res/ss_tot) if ss_tot > 0 else 0.0,
        "mape": float(np.average(np.abs(err / np.clip(actual, 0.01, None)), weights=w) * 100),
        "bias": float(np.average(err, weights=w)),
        "fleet_err_pct": float((pred.sum()/actual.sum() - 1) * 100),
    }


# ---------------------------------------------------------------------------
def main():
    print("Loading data ...")
    with open(os.path.join(cfg.DATA_DIR, "osm_road_cache.pkl"), "rb") as f:
        osm_cache = pickle.load(f)

    train = train_best.add_road_features(
        pd.read_parquet(os.path.join(cfg.DATA_DIR, "train_trips.parquet")), osm_cache)
    val = train_best.add_road_features(
        pd.read_parquet(os.path.join(cfg.DATA_DIR, "val_trips.parquet")), osm_cache)
    test = train_best.add_road_features(
        pd.read_parquet(os.path.join(cfg.DATA_DIR, "test_trips.parquet")), osm_cache)
    for df in [train, val, test]:
        df["energy_whkm"] = np.where(df["distance_m"] > 0,
                                     df["energy_wh"]/(df["distance_m"]/1000.0), np.nan)

    gps_features = [f for f in cfg.GPS_FEATURES if f in test.columns and test[f].notna().any()]
    feature_cols = gps_features + train_best.ROAD_FEATURE_COLS
    print(f"Features: {len(feature_cols)}")

    # Reuse the existing seed-42 P90 model from gps_best/ to avoid retraining
    P90_EXISTING = os.path.join(cfg.MODEL_DIR, "gps_best", "lstm_model_seed42.pt")
    SCALER_PATH  = os.path.join(cfg.MODEL_DIR, "gps_best", "lstm_scaler.joblib")

    # Use a SINGLE scaler across all clip levels (the one already saved by 06c).
    # Same features → same scaler → fair comparison.
    scaler = joblib.load(SCALER_PATH)

    # Pre-scale train and val once
    train_sc = train.copy(); train_sc[feature_cols] = scaler.transform(train[feature_cols].fillna(0))
    val_sc   = val.copy();   val_sc[feature_cols]   = scaler.transform(val[feature_cols].fillna(0))

    results = {}
    for label, q in CLIP_LEVELS:
        print(f"\n{'='*70}\nClip level: {label}\n{'='*70}", flush=True)

        if q is None:
            clip_val = float("inf")
            tr = train_sc.copy(); tr["energy_whkm_clipped"] = train["energy_whkm"].values
            vl = val_sc.copy();   vl["energy_whkm_clipped"] = val["energy_whkm"].values
        else:
            clip_val = float(train["energy_whkm"].quantile(q))
            tr = train_sc.copy(); tr["energy_whkm_clipped"] = train["energy_whkm"].clip(upper=clip_val).values
            vl = val_sc.copy();   vl["energy_whkm_clipped"] = val["energy_whkm"].clip(upper=clip_val).values
        print(f"  Clip value (Wh/km): {clip_val:.2f}", flush=True)

        model_path = os.path.join(OUT_DIR, f"lstm_seed42_{label}.pt")

        if label == "p90" and os.path.exists(P90_EXISTING):
            # Reuse the already-trained seed-42 P90 model
            print(f"  Reusing existing P90 model: {P90_EXISTING}", flush=True)
            model_path = P90_EXISTING
        elif os.path.exists(model_path):
            print(f"  Already trained — skipping training", flush=True)
        else:
            print(f"  Training (~85 min)...", flush=True)
            train_lstm(tr, vl, feature_cols, "energy_whkm_clipped", model_path, SEED)

        # Inference on val + test (use ORIGINAL non-scaled dfs for predict() since it scales internally)
        print(f"  Running inference on val + test ...", flush=True)
        v_pred = predict(model_path, scaler, val,  feature_cols)
        t_pred = predict(model_path, scaler, test, feature_cols)
        v_trip = aggregate_to_trip(val,  v_pred)
        t_trip = aggregate_to_trip(test, t_pred)

        wv = v_trip["n_steps"].values.astype(float)
        wt = t_trip["n_steps"].values.astype(float)

        # Uncalibrated metrics
        unc_v_whkm = metrics_w(v_trip["actual_whkm"].values, (v_trip["pred_wh"]/(v_trip["total_dist"]/1000)).values, wv)
        unc_t_whkm = metrics_w(t_trip["actual_whkm"].values, (t_trip["pred_wh"]/(t_trip["total_dist"]/1000)).values, wt)
        unc_v_wh   = metrics_w(v_trip["actual_wh"].values, v_trip["pred_wh"].values, wv)
        unc_t_wh   = metrics_w(t_trip["actual_wh"].values, t_trip["pred_wh"].values, wt)

        # Calibration factor (val Wh ratio)
        scale = float(v_trip["actual_wh"].sum() / v_trip["pred_wh"].sum())

        # Calibrated metrics on test
        t_cal_wh   = t_trip["pred_wh"].values * scale
        t_cal_whkm = t_cal_wh / (t_trip["total_dist"].values/1000)
        cal_t_whkm = metrics_w(t_trip["actual_whkm"].values, t_cal_whkm, wt)
        cal_t_wh   = metrics_w(t_trip["actual_wh"].values,   t_cal_wh,   wt)

        results[label] = {
            "clip_value": clip_val,
            "scale_factor": scale,
            "n_val_trips": int(len(v_trip)), "n_test_trips": int(len(t_trip)),
            "uncal_val_whkm": unc_v_whkm, "uncal_val_wh": unc_v_wh,
            "uncal_test_whkm": unc_t_whkm, "uncal_test_wh": unc_t_wh,
            "cal_test_whkm": cal_t_whkm, "cal_test_wh": cal_t_wh,
        }

        print(f"  Uncal test:  Wh/km MAE={unc_t_whkm['mae']:.2f}  R²={unc_t_whkm['r2']:.3f}  fleet={unc_t_whkm['fleet_err_pct']:+.2f}%", flush=True)
        print(f"  Calib factor: {scale:.4f}", flush=True)
        print(f"  Cal test:    Wh/km MAE={cal_t_whkm['mae']:.2f}  R²={cal_t_whkm['r2']:.3f}  fleet={cal_t_whkm['fleet_err_pct']:+.2f}%", flush=True)

    out_path = os.path.join(cfg.RESULTS_DIR, "lstm_clip_sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)

    # Print summary table
    print("\n" + "="*120)
    print(f"{'clip':<8} {'clip_val':>9} {'scale':>7}  {'uncal Wh/km MAE':>16} {'uncal R²':>9} {'uncal fleet':>11}  {'cal Wh/km MAE':>14} {'cal R²':>7} {'cal fleet':>9}")
    print("="*120)
    for k, v in results.items():
        print(f"{k:<8} {v['clip_value']:>9.2f} {v['scale_factor']:>7.4f}  "
              f"{v['uncal_test_whkm']['mae']:>16.3f} {v['uncal_test_whkm']['r2']:>9.4f} {v['uncal_test_whkm']['fleet_err_pct']:>+10.2f}%  "
              f"{v['cal_test_whkm']['mae']:>14.3f} {v['cal_test_whkm']['r2']:>7.4f} {v['cal_test_whkm']['fleet_err_pct']:>+8.2f}%")


if __name__ == "__main__":
    main()
