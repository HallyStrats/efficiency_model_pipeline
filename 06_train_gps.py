"""
06_train_gps.py

Trains Model 1: GPS-Only models using only coordinate-derived features.
Three sub-models:
  A. Per-timestep XGBoost → aggregate to trip
  B. Trip-level aggregate XGBoost
  C. Per-timestep LSTM → aggregate to trip

Output:
  models/gps_xgb_timestep.json
  models/gps_xgb_trip.json
  models/gps_lstm.pt
  models/gps_scaler.joblib
"""

import os
import json

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from importlib import import_module
cfg = import_module("00_config")


# ---------------------------------------------------------------------------
# LSTM model
# ---------------------------------------------------------------------------

class TripLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
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
    def __init__(self, trips_df, feature_cols, target_col):
        self.sequences = []
        self.targets = []

        # Fast vectorised split — avoid slow pandas groupby iteration
        X_all = np.nan_to_num(trips_df[feature_cols].values.astype(np.float32), 0.0)
        y_all = np.nan_to_num(trips_df[target_col].values.astype(np.float32), 0.0)
        trip_ids = trips_df["trip_id"].values

        # Find boundaries where trip_id changes
        breaks = np.where(np.diff(trip_ids) != 0)[0] + 1
        starts = np.concatenate([[0], breaks])
        ends = np.concatenate([breaks, [len(trip_ids)]])

        for s, e in zip(starts, ends):
            if e - s > 0:
                self.sequences.append(torch.from_numpy(X_all[s:e]))
                self.targets.append(torch.from_numpy(y_all[s:e]))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def collate_trips(batch):
    seqs, targets = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    seqs_padded = pad_sequence(seqs, batch_first=True)
    targets_padded = pad_sequence(targets, batch_first=True)
    return seqs_padded, targets_padded, lengths


# ---------------------------------------------------------------------------
# Feature preparation helpers
# ---------------------------------------------------------------------------

def get_available_features(df, feature_list):
    """Return features that actually exist and have data."""
    available = []
    for col in feature_list:
        if col in df.columns and df[col].notna().any():
            available.append(col)
    return available


def prepare_timestep_data(train_df, val_df, feature_cols, target_col):
    """Prepare flat X, y arrays for XGBoost from row-level data."""
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df[target_col].values
    X_val = val_df[feature_cols].fillna(0).values
    y_val = val_df[target_col].values

    # Remove NaN targets
    train_valid = ~np.isnan(y_train)
    val_valid = ~np.isnan(y_val)
    return X_train[train_valid], y_train[train_valid], X_val[val_valid], y_val[val_valid]


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_xgb_timestep(train_df, val_df, feature_cols, model_path):
    """Train per-timestep XGBoost model."""
    print("\n" + "="*70)
    print("Model A: GPS Per-timestep XGBoost")
    print("="*70)

    X_train, y_train, X_val, y_val = prepare_timestep_data(
        train_df, val_df, feature_cols, cfg.TARGET_COL
    )
    print(f"  Train: {len(X_train):,} rows × {len(feature_cols)} features")
    print(f"  Val:   {len(X_val):,} rows")

    model = xgb.XGBRegressor(
        **cfg.XGB_PARAMS,
        early_stopping_rounds=cfg.XGB_EARLY_STOPPING,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    model.save_model(model_path)
    print(f"  Saved → {model_path}")

    # Feature importance
    imp = dict(zip(feature_cols, model.feature_importances_))
    imp_sorted = sorted(imp.items(), key=lambda x: -x[1])
    print(f"\n  Top 10 features:")
    for name, score in imp_sorted[:10]:
        print(f"    {name:<30} {score:.4f}")

    return model


def train_xgb_trip(train_sum, val_sum, feature_cols, model_path):
    """Train trip-level aggregate XGBoost model."""
    print("\n" + "="*70)
    print("Model B: GPS Trip-level XGBoost")
    print("="*70)

    target = "trip_energy_wh"
    avail = [c for c in feature_cols if c in train_sum.columns and train_sum[c].notna().any()]

    X_train = train_sum[avail].fillna(0).values
    y_train = train_sum[target].values
    X_val = val_sum[avail].fillna(0).values
    y_val = val_sum[target].values

    valid_train = ~np.isnan(y_train)
    valid_val = ~np.isnan(y_val)
    X_train, y_train = X_train[valid_train], y_train[valid_train]
    X_val, y_val = X_val[valid_val], y_val[valid_val]

    print(f"  Train: {len(X_train):,} trips × {len(avail)} features")
    print(f"  Val:   {len(X_val):,} trips")

    model = xgb.XGBRegressor(
        **cfg.XGB_PARAMS,
        early_stopping_rounds=cfg.XGB_EARLY_STOPPING,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    model.save_model(model_path)
    print(f"  Saved → {model_path}")

    # Save feature names
    meta_path = model_path.replace(".json", "_features.json")
    with open(meta_path, "w") as f:
        json.dump(avail, f, indent=2)

    return model, avail


def train_lstm(train_df, val_df, feature_cols, model_path, scaler_path):
    """Train per-timestep LSTM model."""
    print("\n" + "="*70)
    print("Model C: GPS Per-timestep LSTM")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Device: {device}")

    # Scale features
    scaler = StandardScaler()
    train_features = train_df[feature_cols].fillna(0).values
    scaler.fit(train_features)
    joblib.dump(scaler, scaler_path)

    # Scale the data in-place in copies
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols].fillna(0))
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols].fillna(0))

    train_ds = TripDataset(train_scaled, feature_cols, cfg.TARGET_COL)
    val_ds = TripDataset(val_scaled, feature_cols, cfg.TARGET_COL)
    print(f"  Train sequences: {len(train_ds):,}")
    print(f"  Val sequences:   {len(val_ds):,}")

    if len(train_ds) == 0:
        print("  WARNING: No valid training sequences. Skipping LSTM.")
        return None

    train_loader = DataLoader(train_ds, batch_size=cfg.LSTM_PARAMS["batch_size"],
                              shuffle=True, collate_fn=collate_trips)
    val_loader = DataLoader(val_ds, batch_size=cfg.LSTM_PARAMS["batch_size"],
                            shuffle=False, collate_fn=collate_trips)

    model = TripLSTM(
        input_size=len(feature_cols),
        hidden_size=cfg.LSTM_PARAMS["hidden_size"],
        num_layers=cfg.LSTM_PARAMS["num_layers"],
        dropout=cfg.LSTM_PARAMS["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LSTM_PARAMS["learning_rate"],
        weight_decay=cfg.LSTM_PARAMS["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.HuberLoss(delta=1.0)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(cfg.LSTM_PARAMS["max_epochs"]):
        # Train
        model.train()
        train_loss = 0
        n_batches = 0
        for seqs, targets, lengths in train_loader:
            seqs, targets, lengths = seqs.to(device), targets.to(device), lengths.to(device)
            preds = model(seqs, lengths)
            # Mask padding
            mask = torch.arange(targets.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
            loss = criterion(preds[mask], targets[mask])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        # Validate
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for seqs, targets, lengths in val_loader:
                seqs, targets, lengths = seqs.to(device), targets.to(device), lengths.to(device)
                preds = model(seqs, lengths)
                mask = torch.arange(targets.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
                loss = criterion(preds[mask], targets[mask])
                val_loss += loss.item()
                n_val += 1

        avg_train = train_loss / max(n_batches, 1)
        avg_val = val_loss / max(n_val, 1)
        scheduler.step(avg_val)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.LSTM_PARAMS["patience"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Saved → {model_path}")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)

    # Load data
    train_trips = pd.read_parquet(os.path.join(cfg.DATA_DIR, "train_trips.parquet"))
    val_trips = pd.read_parquet(os.path.join(cfg.DATA_DIR, "val_trips.parquet"))
    train_sum = pd.read_parquet(os.path.join(cfg.DATA_DIR, "train_summaries.parquet"))
    val_sum = pd.read_parquet(os.path.join(cfg.DATA_DIR, "val_summaries.parquet"))

    # Determine available GPS features
    gps_features = get_available_features(train_trips, cfg.GPS_FEATURES)
    print(f"GPS features available: {len(gps_features)}/{len(cfg.GPS_FEATURES)}")
    for f in gps_features:
        print(f"  {f}")

    # GPS trip-level feature columns
    gps_trip_features = [c for c in train_sum.columns
                         if c.startswith("trip_") and c not in
                         ("trip_id", "trip_energy_wh", "trip_wh_per_km", "rider_id",
                          "trip_mean_rpm", "trip_mean_throttle", "trip_brake_ratio",
                          "trip_mean_speed_limit_ratio", "trip_mean_vehicle_speed",
                          "trip_max_vehicle_speed")]

    # --- Model A: Per-timestep XGBoost ---
    train_xgb_timestep(
        train_trips, val_trips, gps_features,
        os.path.join(cfg.MODEL_DIR, "gps_xgb_timestep.json"),
    )

    # --- Model B: Trip-level XGBoost ---
    train_xgb_trip(
        train_sum, val_sum, gps_trip_features,
        os.path.join(cfg.MODEL_DIR, "gps_xgb_trip.json"),
    )

    # --- Model C: LSTM ---
    train_lstm(
        train_trips, val_trips, gps_features,
        os.path.join(cfg.MODEL_DIR, "gps_lstm.pt"),
        os.path.join(cfg.MODEL_DIR, "gps_scaler.joblib"),
    )

    print("\n" + "="*70)
    print("GPS-only model training complete.")
    print("="*70)


if __name__ == "__main__":
    main()
