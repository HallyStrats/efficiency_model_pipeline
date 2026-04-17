"""
02_clean.py

Reads raw_ingested.parquet, applies data quality filters, smooths the rc
sensor signal, and outputs a clean dataset ready for feature engineering.

Steps:
  1. Drop GPS dropout rows (lat or long == 0)
  2. Drop riders with no BMS data (all rc null)
  3. Apply 3-point median filter to rc per rider (target noise suppression)
  4. Interpolate single-row BMS nulls (limit=1)
  5. Drop rc outside valid range
  6. Drop cell voltages outside valid range
  7. Drop remaining rows with null rc
  8. Clean sensor columns (heading -1 → NaN, odometer -1 → NaN)

Output: data/cleaned.parquet
"""

import os

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from importlib import import_module
cfg = import_module("00_config")


def main():
    in_path = os.path.join(cfg.DATA_DIR, "raw_ingested.parquet")
    print(f"Reading {in_path} ...")
    df = pd.read_parquet(in_path)
    print(f"Loaded: {len(df):,} rows × {len(df.columns)} columns\n")

    # --- 1. Drop GPS dropout rows ---
    before = len(df)
    gps_valid = (df["lat"] != 0.0) & (df["long"] != 0.0) & df["lat"].notna() & df["long"].notna()
    df = df[gps_valid].reset_index(drop=True)
    print(f"[1] GPS dropout removed:          {before - len(df):>8,}  → {len(df):,} rows")

    # --- 2. Drop riders with no BMS data ---
    rider_rc_counts = df.groupby("rider_id")["rc"].apply(lambda s: s.notna().sum())
    valid_riders = rider_rc_counts[rider_rc_counts > 0].index
    before = len(df)
    n_riders_before = df["rider_id"].nunique()
    df = df[df["rider_id"].isin(valid_riders)].reset_index(drop=True)
    n_riders_after = df["rider_id"].nunique()
    print(f"[2] Riders with no BMS dropped:    {n_riders_before - n_riders_after} riders, "
          f"{before - len(df):>8,} rows  → {len(df):,} rows ({n_riders_after} riders)")

    # --- 3. Median filter on rc per rider (suppress sensor spikes) ---
    print(f"[3] Applying median filter (w={cfg.RC_MEDIAN_WINDOW}) to rc per rider ...")
    def smooth_rc(group):
        rc = group["rc"].values.copy()
        valid = ~np.isnan(rc)
        if valid.sum() > cfg.RC_MEDIAN_WINDOW:
            rc_valid = rc[valid]
            rc_valid = median_filter(rc_valid, size=cfg.RC_MEDIAN_WINDOW)
            rc[valid] = rc_valid
        group = group.copy()
        group["rc"] = rc
        return group

    df = df.groupby("rider_id", group_keys=False).apply(smooth_rc).reset_index(drop=True)
    print(f"     rc median filter applied.")

    # --- 4. Interpolate single-row BMS nulls ---
    bms_cols = ["rc"] + [c for c in cfg.CELL_COLS if c in df.columns]
    before_nulls = df[bms_cols].isna().sum().sum()
    for col in bms_cols:
        df[col] = df.groupby("rider_id")[col].transform(
            lambda s: s.interpolate(method="linear", limit=1, limit_direction="forward")
        )
    after_nulls = df[bms_cols].isna().sum().sum()
    print(f"[4] BMS interpolation (limit=1):   {before_nulls - after_nulls:>8,} nulls filled")

    # --- 5. Drop rc outside valid range ---
    before = len(df)
    df = df[(df["rc"].isna()) | ((df["rc"] > 0) & (df["rc"] <= cfg.RC_MAX_MAH))]
    print(f"[5] rc out of range removed:       {before - len(df):>8,}  → {len(df):,} rows")

    # --- 6. Drop rows with cell voltages outside valid range ---
    existing_cells = [c for c in cfg.CELL_COLS if c in df.columns]
    if existing_cells:
        before = len(df)
        for col in existing_cells:
            mask = df[col].notna() & ((df[col] < cfg.CELL_V_MIN_MV) | (df[col] > cfg.CELL_V_MAX_MV))
            df.loc[mask, col] = np.nan
        # Don't drop rows — just null out bad cell values
        print(f"[6] Bad cell voltages nulled out")

    # --- 7. Drop rows with null rc (needed for target) ---
    before = len(df)
    df = df.dropna(subset=["rc"]).reset_index(drop=True)
    print(f"[7] Null rc rows dropped:          {before - len(df):>8,}  → {len(df):,} rows")

    # --- 8. Clean sensor columns ---
    if "heading" in df.columns:
        df.loc[df["heading"] == -1, "heading"] = np.nan
    if "odometer" in df.columns:
        df.loc[df["odometer"] == -1, "odometer"] = np.nan
    # Convert is_ignition_on to boolean
    if "is_ignition_on" in df.columns:
        df["is_ignition_on"] = df["is_ignition_on"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
    print(f"[8] Sensor columns cleaned\n")

    # --- Save ---
    out_path = os.path.join(cfg.DATA_DIR, "cleaned.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Saved: {len(df):,} rows × {len(df.columns)} columns → {out_path}")

    # Summary
    print(f"\nColumn completeness:")
    for col in df.columns:
        pct = df[col].notna().mean() * 100
        print(f"  {col:<25} {pct:5.1f}%")


if __name__ == "__main__":
    main()
