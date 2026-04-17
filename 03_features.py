"""
03_features.py

Reads cleaned.parquet, engineers ALL features for both models (GPS-derived
and sensor-derived), computes the energy target, applies quality filters,
and writes features.parquet.

GPS-derived features (Model 1):
  speed_ms, accel_ms2, jerk_ms3, distance_m, elevation_m, slope_pct,
  heading_sin, heading_cos, heading_change_rate, curvature,
  hour_sin, hour_cos, day_of_week, dt_seconds,
  speed_x_slope, kinetic_energy_proxy

Sensor-derived features (Model 2 adds):
  vehicle_speed, spd, rpm, ts, bm, heading_device, prnd, gr,
  road_speed_limit, speed_limit_ratio, throttle_x_speed,
  accel_x_throttle, rpm_rolling_mean, throttle_rolling_std,
  is_braking_while_fast

Target:
  energy_wh = (rc[t-1] - rc[t]) × mean_pack_voltage_V / 1000

Output: data/features.parquet
"""

import os
import math

import numpy as np
import pandas as pd
import srtm
from scipy.ndimage import median_filter

from importlib import import_module
cfg = import_module("00_config")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def haversine_series(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """Vectorised Haversine between consecutive rows. First row → NaN."""
    R = 6_371_000.0
    lat_r = np.radians(lat.values.astype(float))
    lon_r = np.radians(lon.values.astype(float))
    dphi = np.diff(lat_r)
    dlam = np.diff(lon_r)
    phi1, phi2 = lat_r[:-1], lat_r[1:]
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    dist = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return pd.Series(np.concatenate([[np.nan], dist]), index=lat.index)


def lookup_srtm_elevations(df: pd.DataFrame) -> pd.Series:
    """SRTM elevation for every (lat, lon). Rounded coords for cache efficiency."""
    srtm_data = srtm.get_data()
    lats = df["lat"].round(cfg.SRTM_ROUND_DP)
    lons = df["long"].round(cfg.SRTM_ROUND_DP)

    valid = lats.notna() & lons.notna()
    unique = pd.DataFrame({"lat": lats[valid], "lon": lons[valid]}).drop_duplicates()

    elev_map = {}
    total = len(unique)
    for i, (_, row) in enumerate(unique.iterrows()):
        key = (row["lat"], row["lon"])
        elev_map[key] = srtm_data.get_elevation(float(row["lat"]), float(row["lon"]))
        if (i + 1) % 5000 == 0:
            print(f"     SRTM lookup: {i+1:,}/{total:,}")

    print(f"     SRTM lookup complete: {total:,} unique coordinates")

    result = [
        elev_map.get((la, lo), np.nan) if (pd.notna(la) and pd.notna(lo)) else np.nan
        for la, lo in zip(lats, lons)
    ]
    return pd.Series(result, index=df.index, dtype="float64")


def circular_diff(angles: pd.Series) -> pd.Series:
    """Circular difference in radians, wrapping to [-π, π]."""
    d = angles.diff()
    return ((d + np.pi) % (2 * np.pi)) - np.pi


# ---------------------------------------------------------------------------
# Per-rider feature computation
# ---------------------------------------------------------------------------

def compute_features_for_rider(rdf: pd.DataFrame) -> pd.DataFrame:
    """Compute all kinematic and sensor features for one rider's data."""
    df = rdf.copy()

    # --- Time delta ---
    dt_raw = df["gps_date"].diff().dt.total_seconds()
    # NaN at session boundaries and zero-dt rows
    session_break = (dt_raw > cfg.SESSION_GAP_S) | (dt_raw == 0)
    df["dt_seconds"] = dt_raw.where(~session_break)

    # =====================================================================
    # GPS-DERIVED FEATURES (Model 1)
    # =====================================================================

    # --- Median filter on coordinates (suppress GPS outliers) ---
    lat_arr = df["lat"].values.copy()
    lon_arr = df["long"].values.copy()
    valid_gps = ~(np.isnan(lat_arr) | np.isnan(lon_arr))
    if valid_gps.sum() > 3:
        lat_arr[valid_gps] = median_filter(lat_arr[valid_gps], size=3)
        lon_arr[valid_gps] = median_filter(lon_arr[valid_gps], size=3)
    df["lat_smooth"] = lat_arr
    df["lon_smooth"] = lon_arr

    # --- Step distance (haversine from smoothed coords) ---
    df["distance_m"] = haversine_series(df["lat_smooth"], df["lon_smooth"])
    # NaN at session boundaries
    df.loc[session_break, "distance_m"] = np.nan

    # --- Speed from coordinates ---
    df["speed_ms"] = df["distance_m"] / df["dt_seconds"]
    df.loc[df["speed_ms"] > cfg.SPEED_CAP_MS, "speed_ms"] = np.nan

    # --- Acceleration ---
    df["accel_ms2"] = df["speed_ms"].diff() / df["dt_seconds"]
    df["accel_ms2"] = df["accel_ms2"].clip(-cfg.ACCEL_CAP_MS2, cfg.ACCEL_CAP_MS2)
    df.loc[session_break, "accel_ms2"] = np.nan

    # --- Jerk ---
    df["jerk_ms3"] = df["accel_ms2"].diff() / df["dt_seconds"]
    df.loc[session_break, "jerk_ms3"] = np.nan

    # --- Heading from coordinate deltas ---
    dlat = df["lat_smooth"].diff()
    dlon = df["lon_smooth"].diff()
    heading_rad = np.arctan2(dlon, dlat)
    df["heading_sin"] = np.sin(heading_rad)
    df["heading_cos"] = np.cos(heading_rad)
    df.loc[session_break, ["heading_sin", "heading_cos"]] = np.nan

    # --- Heading change rate ---
    heading_diff = circular_diff(heading_rad)
    df["heading_change_rate"] = heading_diff / df["dt_seconds"]
    df.loc[session_break, "heading_change_rate"] = np.nan

    # --- Curvature = heading_change_rate / speed ---
    df["curvature"] = np.where(
        df["speed_ms"] > cfg.MIN_SPEED_MS,
        df["heading_change_rate"] / df["speed_ms"],
        0.0,
    )

    # --- Temporal features ---
    df["hour_sin"] = np.sin(2 * np.pi * df["gps_date"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["gps_date"].dt.hour / 24)
    df["day_of_week"] = df["gps_date"].dt.dayofweek

    # --- Interaction features ---
    # slope_pct is computed after SRTM lookup, so we'll fill these in main()
    df["kinetic_energy_proxy"] = 0.5 * df["speed_ms"] ** 2

    # =====================================================================
    # SENSOR-DERIVED FEATURES (Model 2)
    # =====================================================================

    # Copy raw sensor columns, renaming heading to avoid conflict
    if "heading" in df.columns:
        df["heading_device"] = df["heading"]

    # Speed limit ratio
    if "road_speed_limit" in df.columns and "vehicle_speed" in df.columns:
        rsl = df["road_speed_limit"].replace(0, np.nan)
        df["speed_limit_ratio"] = df["vehicle_speed"] / rsl

    # Throttle × speed interaction
    if "ts" in df.columns and "vehicle_speed" in df.columns:
        df["throttle_x_speed"] = df["ts"].fillna(0) * df["vehicle_speed"].fillna(0) / 100.0

    # Acceleration × throttle
    if "ts" in df.columns:
        df["accel_x_throttle"] = df["accel_ms2"].fillna(0) * df["ts"].fillna(0) / 100.0

    # Rolling RPM mean
    if "rpm" in df.columns:
        df["rpm_rolling_mean"] = df["rpm"].rolling(5, min_periods=1, center=True).mean()

    # Rolling throttle std
    if "ts" in df.columns:
        df["throttle_rolling_std"] = df["ts"].rolling(5, min_periods=1, center=True).std().fillna(0)

    # Braking while fast
    if "bm" in df.columns and "vehicle_speed" in df.columns:
        df["is_braking_while_fast"] = ((df["bm"] == 1) & (df["vehicle_speed"] > 20)).astype(float)

    # Drop smoothed intermediates
    df = df.drop(columns=["lat_smooth", "lon_smooth"], errors="ignore")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    in_path = os.path.join(cfg.DATA_DIR, "cleaned.parquet")
    print(f"Reading {in_path} ...")
    df = pd.read_parquet(in_path)
    print(f"Loaded: {len(df):,} rows × {len(df.columns)} columns\n")

    # --- Compute per-rider features ---
    print("Computing per-rider kinematic features ...")
    riders = sorted(df["rider_id"].unique())
    processed = []
    for i, rid in enumerate(riders):
        rdf = df[df["rider_id"] == rid].copy()
        rdf = compute_features_for_rider(rdf)
        processed.append(rdf)
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(riders)} riders")
    df = pd.concat(processed, ignore_index=True)
    print(f"  Processed {len(riders)} riders total\n")

    # --- SRTM elevation lookup ---
    print("Looking up SRTM elevations ...")
    df["elevation_m"] = lookup_srtm_elevations(df)

    # --- Road grade from elevation ---
    # Need to compute per rider to avoid cross-rider contamination
    print("Computing road grade ...")
    slope_list = []
    for rid in riders:
        mask = df["rider_id"] == rid
        rdf = df.loc[mask]
        elev_diff = rdf["elevation_m"].diff()
        dist = rdf["distance_m"]
        slope = (elev_diff / dist.replace(0, np.nan)) * 100.0
        slope = slope.clip(-cfg.SLOPE_CAP_PCT, cfg.SLOPE_CAP_PCT)
        slope_list.append(slope)
    df["slope_pct"] = pd.concat(slope_list)

    # --- Fill interaction feature that needed slope ---
    df["speed_x_slope"] = df["speed_ms"] * df["slope_pct"]

    # =====================================================================
    # ENERGY TARGET
    # =====================================================================
    print("\nComputing energy target ...")

    # Pack voltage from cell voltages
    existing_cells = [c for c in cfg.CELL_COLS if c in df.columns]
    if existing_cells:
        # Mean cell voltage → pack voltage (cells in series)
        cell_mean_mv = df[existing_cells].mean(axis=1, skipna=True)
        # Where all cells are null, use nominal
        cell_mean_mv = cell_mean_mv.fillna(cfg.NOMINAL_CELL_MV)
        pack_voltage_v = cell_mean_mv * cfg.N_SERIES_CELLS / 1000.0
    else:
        pack_voltage_v = cfg.NOMINAL_CELL_MV * cfg.N_SERIES_CELLS / 1000.0

    # Energy per timestep: delta_rc (mAh) × pack_voltage (V) / 1000 → Wh
    # Positive energy_wh means energy was consumed (discharge)
    rc_delta = df.groupby("rider_id")["rc"].diff()
    # Average pack voltage over the step
    if isinstance(pack_voltage_v, pd.Series):
        avg_voltage = (pack_voltage_v + pack_voltage_v.shift(1)) / 2
    else:
        avg_voltage = pack_voltage_v

    df["energy_wh"] = (-rc_delta) * avg_voltage / 1000.0

    # =====================================================================
    # FILTERING
    # =====================================================================
    print("Applying filters ...")

    # Drop first row per rider (no delta)
    before = len(df)
    first_rows = df.groupby("rider_id").head(1).index
    df = df.drop(first_rows).reset_index(drop=True)
    print(f"  First rows per rider:            {before - len(df):>8,}  → {len(df):,}")

    # Drop ignition off
    if "is_ignition_on" in df.columns:
        before = len(df)
        df = df[df["is_ignition_on"] == True].reset_index(drop=True)
        print(f"  Ignition off:                    {before - len(df):>8,}  → {len(df):,}")

    # Drop irregular timesteps
    before = len(df)
    df = df[(df["dt_seconds"] >= cfg.DT_MIN_S) & (df["dt_seconds"] <= cfg.DT_MAX_S)]
    print(f"  dt outside [{cfg.DT_MIN_S}, {cfg.DT_MAX_S}]s:              {before - len(df):>8,}  → {len(df):,}")

    # Drop stationary rows
    before = len(df)
    moving = df["speed_ms"].notna() & (df["speed_ms"] > cfg.MIN_SPEED_MS)
    df = df[moving].reset_index(drop=True)
    print(f"  Stationary (speed < {cfg.MIN_SPEED_MS}):       {before - len(df):>8,}  → {len(df):,}")

    # Drop low elevation (GPS anomalies below Nairobi)
    before = len(df)
    df = df[(df["elevation_m"].isna()) | (df["elevation_m"] >= cfg.ELEVATION_MIN_M)]
    print(f"  Elevation < {cfg.ELEVATION_MIN_M}m:              {before - len(df):>8,}  → {len(df):,}")

    # Drop invalid energy values
    before = len(df)
    # Only keep positive discharge
    valid_energy = (df["energy_wh"] > 0) & (df["energy_wh"].notna())
    # Cap based on power: energy_wh / (dt/3600) < DISCHARGE_MAX_W
    power_w = df["energy_wh"] / (df["dt_seconds"] / 3600.0)
    valid_energy = valid_energy & (power_w <= cfg.DISCHARGE_MAX_W)
    # Cap single-step rc change
    valid_energy = valid_energy & ((-rc_delta.reindex(df.index)).abs() <= cfg.RC_STEP_MAX_MAH)
    df = df[valid_energy].reset_index(drop=True)
    print(f"  Invalid energy:                  {before - len(df):>8,}  → {len(df):,}")

    # Drop rows with null speed (needed for both models)
    before = len(df)
    df = df.dropna(subset=["speed_ms", "distance_m", "dt_seconds", "energy_wh"]).reset_index(drop=True)
    print(f"  Null core features:              {before - len(df):>8,}  → {len(df):,}")

    # =====================================================================
    # DROP TARGET-DERIVATION COLUMNS (never used as features)
    # =====================================================================
    drop_target_cols = ["rc", "fc", "is_ignition_on"] + [c for c in cfg.CELL_COLS if c in df.columns]
    df = df.drop(columns=[c for c in drop_target_cols if c in df.columns], errors="ignore")

    # =====================================================================
    # SAVE
    # =====================================================================
    out_path = os.path.join(cfg.DATA_DIR, "features.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\nSaved: {len(df):,} rows × {len(df.columns)} columns → {out_path}")

    # Feature availability summary
    print(f"\nGPS features availability:")
    for col in cfg.GPS_FEATURES:
        if col in df.columns:
            pct = df[col].notna().mean() * 100
            print(f"  {col:<25} {pct:5.1f}%")
        else:
            print(f"  {col:<25} MISSING")

    print(f"\nSensor features availability:")
    for col in cfg.SENSOR_FEATURES:
        if col in df.columns:
            pct = df[col].notna().mean() * 100
            print(f"  {col:<25} {pct:5.1f}%")
        else:
            print(f"  {col:<25} MISSING")

    print(f"\n  {cfg.TARGET_COL:<25} {df[cfg.TARGET_COL].notna().mean()*100:5.1f}%")


if __name__ == "__main__":
    main()
