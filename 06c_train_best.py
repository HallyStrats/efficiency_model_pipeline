"""
06c_train_best.py

Trains the best GPS-only model found through experimentation:
  - LSTM with 23 features (16 GPS-derived + 7 OSM road type)
  - Target: Wh/km per timestep, clipped at P90 to reduce target noise
  - Also trains XGBoost for comparison

This script encapsulates all the decisions that led to the best Wh/km R²:
  1. OSM road type features (ordinal + one-hot)
  2. Wh/km target instead of raw Wh (removes trip-length dependency)
  3. P90 target clipping (reduces influence of noisy extreme values)
  4. 2-layer LSTM (captures sequential driving patterns XGBoost cannot)

Also trains an Optuna-tuned XGBoost as a baseline comparison.

Output:
  models/gps_best/lstm_model.pt
  models/gps_best/lstm_scaler.joblib
  models/gps_best/xgb_model.json
  models/gps_best/xgb_params.json
  models/gps_best/metadata.json
"""

import json
import os
import pickle
from datetime import datetime

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                 pad_sequence)
from torch.utils.data import DataLoader, Dataset

from importlib import import_module
cfg = import_module("00_config")

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Road type encoding
# ---------------------------------------------------------------------------

ROAD_ORD = {
    "unknown": 0, "living_street": 1, "residential": 1,
    "unclassified": 2, "tertiary_link": 2, "tertiary": 3,
    "secondary_link": 3, "secondary": 4,
    "primary_link": 4, "primary": 5,
    "trunk_link": 5, "trunk": 6,
    "motorway_link": 6, "motorway": 7,
}

ROAD_ONEHOT_CATS = ["residential", "secondary", "tertiary", "primary", "trunk", "motorway"]

ROAD_FEATURE_COLS = ["road_type_ord"] + [f"road_{c}" for c in ROAD_ONEHOT_CATS]


def add_road_features(df, osm_cache):
    """Add road type ordinal + one-hot features from OSM cache."""
    df = df.copy()
    coords = list(zip(np.round(df["lat"].values, 4), np.round(df["long"].values, 4)))
    road_types = [osm_cache.get(c, "unknown") for c in coords]

    # Simplify link types
    simplified = []
    for rt in road_types:
        rt_clean = rt.replace("_link", "") if rt.endswith("_link") else rt
        if rt_clean == "living_street":
            rt_clean = "residential"
        simplified.append(rt_clean)

    df["road_type_ord"] = [ROAD_ORD.get(rt, 0) for rt in road_types]
    for cat in ROAD_ONEHOT_CATS:
        df[f"road_{cat}"] = [1.0 if s == cat else 0.0 for s in simplified]
    return df


# ---------------------------------------------------------------------------
# LSTM model and dataset
# ---------------------------------------------------------------------------

class TripLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.head(out).squeeze(-1)


class TripDataset(Dataset):
    def __init__(self, df, feature_cols, target_col):
        self.sequences, self.targets = [], []
        X = np.nan_to_num(df[feature_cols].values.astype(np.float32), 0.0)
        y = np.nan_to_num(df[target_col].values.astype(np.float32), 0.0)
        ids = df["trip_id"].values
        breaks = np.where(np.diff(ids) != 0)[0] + 1
        starts = np.concatenate([[0], breaks])
        ends = np.concatenate([breaks, [len(ids)]])
        for s, e in zip(starts, ends):
            if e - s > 0:
                self.sequences.append(torch.from_numpy(X[s:e]))
                self.targets.append(torch.from_numpy(y[s:e]))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def collate_trips(batch):
    seqs, targets = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    return pad_sequence(seqs, batch_first=True), pad_sequence(targets, batch_first=True), lengths


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_lstm(train_df, val_df, feature_cols, target_col, model_path, scaler_path):
    """Train 2-layer LSTM on per-timestep Wh/km."""
    print("\n" + "=" * 70)
    print("Training LSTM (GPS-only, Wh/km target)")
    print("=" * 70)

    device = torch.device("cpu")  # MPS can hang; CPU is reliable
    print(f"  Device: {device}")

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].fillna(0).values)
    joblib.dump(scaler, scaler_path)

    train_sc = train_df.copy()
    val_sc = val_df.copy()
    train_sc[feature_cols] = scaler.transform(train_df[feature_cols].fillna(0))
    val_sc[feature_cols] = scaler.transform(val_df[feature_cols].fillna(0))

    train_ds = TripDataset(train_sc, feature_cols, target_col)
    val_ds = TripDataset(val_sc, feature_cols, target_col)
    print(f"  Train: {len(train_ds):,} trips")
    print(f"  Val:   {len(val_ds):,} trips")

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_trips)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_trips)

    model = TripLSTM(input_size=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.HuberLoss(delta=1.0)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(100):
        model.train()
        train_loss, n_b = 0, 0
        for seqs, targets, lengths in train_loader:
            seqs, targets = seqs.to(device), targets.to(device)
            preds = model(seqs, lengths)
            mask = torch.arange(targets.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
            loss = criterion(preds[mask], targets[mask])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_b += 1

        model.eval()
        val_loss, n_v = 0, 0
        with torch.no_grad():
            for seqs, targets, lengths in val_loader:
                seqs, targets = seqs.to(device), targets.to(device)
                preds = model(seqs, lengths)
                mask = torch.arange(targets.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
                val_loss += criterion(preds[mask], targets[mask]).item()
                n_v += 1

        avg_train = train_loss / max(n_b, 1)
        avg_val = val_loss / max(n_v, 1)
        scheduler.step(avg_val)

        if (epoch + 1) % 5 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:3d}: train={avg_train:.4f}  val={avg_val:.4f}  lr={lr:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Saved → {model_path}")
    return model


def tune_and_train_xgb(train_df, val_df, feature_cols, target_col, model_path, params_path,
                        n_trials=50):
    """Optuna-tune XGBoost on per-timestep Wh/km, evaluated at trip-level MAE."""
    print("\n" + "=" * 70)
    print(f"Tuning XGBoost ({n_trials} Optuna trials)")
    print("=" * 70)

    X_tr = train_df[feature_cols].fillna(0).values
    y_tr = train_df[target_col].values
    X_va = val_df[feature_cols].fillna(0).values
    y_va = val_df[target_col].values
    m_tr = ~np.isnan(y_tr) & np.isfinite(y_tr)
    m_va = ~np.isnan(y_va) & np.isfinite(y_va)
    X_tr, y_tr = X_tr[m_tr], y_tr[m_tr]
    X_va, y_va = X_va[m_va], y_va[m_va]

    # Precompute validation trip actual energy for objective
    val_mask = val_df[target_col].notna() & np.isfinite(val_df[target_col])
    val_trip_actual_wh = val_df[val_mask].groupby("trip_id")["energy_wh"].sum()
    val_trip_ids = val_df.loc[val_mask, "trip_id"].values
    val_X_full = val_df.loc[val_mask, feature_cols].fillna(0).values
    val_dists = val_df.loc[val_mask, "distance_m"].values

    def objective(trial):
        params = {
            "n_estimators": 1500,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
            "max_bin": trial.suggest_int("max_bin", 128, 512),
            "tree_method": "hist",
            "random_state": cfg.SEED,
            "early_stopping_rounds": 30,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=0)

        preds_whkm = model.predict(val_X_full)
        pred_wh = preds_whkm * (val_dists / 1000.0)
        pred_df = pd.DataFrame({"trip_id": val_trip_ids, "pred_wh": pred_wh})
        pred_per_trip = pred_df.groupby("trip_id")["pred_wh"].sum()
        common = val_trip_actual_wh.index.intersection(pred_per_trip.index)
        return mean_absolute_error(val_trip_actual_wh[common], pred_per_trip[common])

    study = optuna.create_study(study_name="gps_best_xgb", direction="minimize")
    for i in range(n_trials):
        study.optimize(objective, n_trials=1)
        if (i + 1) % 10 == 0:
            print(f"  Trial {i+1}/{n_trials}: best MAE = {study.best_value:.2f} Wh")

    best_params = study.best_params
    print(f"\n  Best params: {best_params}")
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    # Retrain with best params
    final_params = {
        **best_params, "n_estimators": 1500, "tree_method": "hist",
        "random_state": cfg.SEED, "early_stopping_rounds": 30,
    }
    model = xgb.XGBRegressor(**final_params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=0)
    model.save_model(model_path)
    print(f"  Saved → {model_path} (best_iteration={model.best_iteration})")
    return model, best_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = os.path.join(cfg.MODEL_DIR, "gps_best")
    os.makedirs(out_dir, exist_ok=True)

    # Load OSM cache
    cache_path = os.path.join(cfg.DATA_DIR, "osm_road_cache.pkl")
    with open(cache_path, "rb") as f:
        osm_cache = pickle.load(f)
    print(f"OSM cache: {len(osm_cache):,} entries")

    # Load data
    print("Loading data...")
    train = add_road_features(pd.read_parquet(os.path.join(cfg.DATA_DIR, "train_trips.parquet")), osm_cache)
    val = add_road_features(pd.read_parquet(os.path.join(cfg.DATA_DIR, "val_trips.parquet")), osm_cache)

    # Build feature list: base GPS + OSM road type
    gps_features = [f for f in cfg.GPS_FEATURES if f in train.columns and train[f].notna().any()]
    feature_cols = gps_features + ROAD_FEATURE_COLS
    print(f"Features: {len(feature_cols)} ({len(gps_features)} GPS + {len(ROAD_FEATURE_COLS)} OSM)")

    # Compute Wh/km target with P90 clip
    for df in [train, val]:
        df["energy_whkm"] = np.where(
            df["distance_m"] > 0,
            df["energy_wh"] / (df["distance_m"] / 1000.0),
            np.nan,
        )
    clip_val = train["energy_whkm"].quantile(0.90)
    print(f"Wh/km P90 clip value: {clip_val:.1f}")

    target = "energy_whkm_clipped"
    train[target] = train["energy_whkm"].clip(upper=clip_val)
    val[target] = val["energy_whkm"].clip(upper=clip_val)

    # --- Train LSTM ---
    lstm_model = train_lstm(
        train, val, feature_cols, target,
        os.path.join(out_dir, "lstm_model.pt"),
        os.path.join(out_dir, "lstm_scaler.joblib"),
    )

    # --- Tune + train XGBoost ---
    xgb_model, xgb_params = tune_and_train_xgb(
        train, val, feature_cols, target,
        os.path.join(out_dir, "xgb_model.json"),
        os.path.join(out_dir, "xgb_params.json"),
        n_trials=50,
    )

    # --- Save metadata ---
    meta = {
        "created": datetime.now().isoformat(),
        "description": "Best GPS-only model: LSTM with 23 features (16 GPS + 7 OSM road type)",
        "features": feature_cols,
        "n_features": len(feature_cols),
        "target": f"energy_whkm (clipped at P90 = {clip_val:.1f})",
        "clip_value": clip_val,
        "lstm": {
            "architecture": "2-layer LSTM, hidden=128, dropout=0.2, HuberLoss",
            "device": "cpu",
            "batch_size": 128,
            "max_epochs": 100,
            "patience": 15,
        },
        "xgb": {
            "tuning": f"{50} Optuna trials",
            "best_params": xgb_params,
        },
        "notes": [
            "GPS-only: uses only lat/long/timestamp + SRTM elevation + OSM road type",
            "Applicable to ICE motorcycles with GPS tracking",
            "LSTM outperforms XGBoost on Wh/km prediction",
            "P90 target clipping reduces influence of noisy extreme values",
        ],
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 70)
    print("Best GPS model training complete.")
    print(f"  Output: {out_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
