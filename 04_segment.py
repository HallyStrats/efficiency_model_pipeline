"""
04_segment.py

Reads features.parquet, segments data into micro-trips based on stationary
gaps, computes per-trip aggregates, and outputs both row-level and trip-level
datasets.

Segmentation:
  1. Session boundaries: dt > 300s or rider change
  2. Micro-trip boundaries: ≥30s cumulative stationary gap within a session

Output:
  data/trips.parquet          — row-level data with trip_id
  data/trip_summaries.parquet — one row per trip with aggregate features
"""

import os

import numpy as np
import pandas as pd

from importlib import import_module
cfg = import_module("00_config")


def segment_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Assign session_id and trip_id to each row.

    Since stationary rows were already removed in 03_features.py, we detect
    micro-trip boundaries by looking at TIME GAPS between consecutive rows.
    A gap > MICROTRIP_GAP_S means there was a stationary period (now removed)
    between two moving segments — that's a micro-trip boundary.
    """
    df = df.copy()

    # Recompute actual time gaps between consecutive rows per rider
    # (dt_seconds from features reflects the original telemetry interval,
    #  but rows may have been removed, creating larger real gaps)
    real_dt = df.groupby("rider_id")["gps_date"].diff().dt.total_seconds()

    # --- Session boundaries ---
    new_rider = df["rider_id"] != df["rider_id"].shift(1)
    long_gap = real_dt > cfg.SESSION_GAP_S
    session_break = new_rider | long_gap
    df["session_id"] = session_break.cumsum()

    # --- Micro-trip boundaries ---
    # Any gap > MICROTRIP_GAP_S between consecutive (filtered) rows
    # indicates a stationary period that was removed → trip boundary
    micro_gap = real_dt > cfg.MICROTRIP_GAP_S
    trip_break = session_break | micro_gap
    df["trip_id"] = trip_break.cumsum()

    return df


def compute_trip_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-trip aggregate features for both model tracks."""

    def agg_trip(tdf):
        n = len(tdf)
        duration = tdf["dt_seconds"].sum()
        distance = tdf["distance_m"].sum()

        result = {
            "rider_id":              tdf["rider_id"].iloc[0],
            "trip_n_steps":          n,
            "trip_duration_s":       duration,
            "trip_distance_m":       distance,
            # GPS-derived aggregates
            "trip_mean_speed":       tdf["speed_ms"].mean(),
            "trip_std_speed":        tdf["speed_ms"].std() if n > 1 else 0,
            "trip_max_speed":        tdf["speed_ms"].max(),
            "trip_mean_accel":       tdf["accel_ms2"].abs().mean(),
            "trip_std_accel":        tdf["accel_ms2"].std() if n > 1 else 0,
            "trip_mean_jerk":        tdf["jerk_ms3"].abs().mean() if "jerk_ms3" in tdf else np.nan,
            "trip_mean_heading_change": tdf["heading_change_rate"].abs().mean()
                                        if "heading_change_rate" in tdf else np.nan,
            "trip_mean_curvature":   tdf["curvature"].abs().mean() if "curvature" in tdf else np.nan,
            # Elevation
            "trip_mean_elevation":   tdf["elevation_m"].mean() if "elevation_m" in tdf else np.nan,
            "trip_mean_slope":       tdf["slope_pct"].mean() if "slope_pct" in tdf else np.nan,
            "trip_max_slope":        tdf["slope_pct"].abs().max() if "slope_pct" in tdf else np.nan,
            # Temporal
            "trip_hour_sin":         tdf["hour_sin"].mean() if "hour_sin" in tdf else np.nan,
            "trip_hour_cos":         tdf["hour_cos"].mean() if "hour_cos" in tdf else np.nan,
            "trip_day_of_week":      tdf["day_of_week"].mode().iloc[0] if "day_of_week" in tdf and len(tdf["day_of_week"].mode()) > 0 else np.nan,
            # Target
            "trip_energy_wh":        tdf[cfg.TARGET_COL].sum(),
        }

        # Elevation gain/loss
        if "elevation_m" in tdf.columns:
            elev_diff = tdf["elevation_m"].diff()
            result["trip_elevation_gain_m"] = elev_diff[elev_diff > 0].sum()
            result["trip_elevation_loss_m"] = (-elev_diff[elev_diff < 0]).sum()
        else:
            result["trip_elevation_gain_m"] = np.nan
            result["trip_elevation_loss_m"] = np.nan

        # Internal stops count (speed drops below threshold within trip)
        speed_below = tdf["speed_ms"] < cfg.MIN_SPEED_MS
        result["trip_n_stops"] = (speed_below & ~speed_below.shift(1, fill_value=False)).sum()

        # Sensor aggregates (Model 2)
        for col, key in [("rpm", "trip_mean_rpm"), ("ts", "trip_mean_throttle"),
                         ("bm", "trip_brake_ratio")]:
            if col in tdf.columns and tdf[col].notna().any():
                if col == "bm":
                    result[key] = tdf[col].mean()  # fraction of time braking
                else:
                    result[key] = tdf[col].mean()
            else:
                result[key] = np.nan

        if "speed_limit_ratio" in tdf.columns and tdf["speed_limit_ratio"].notna().any():
            result["trip_mean_speed_limit_ratio"] = tdf["speed_limit_ratio"].mean()
        else:
            result["trip_mean_speed_limit_ratio"] = np.nan

        if "vehicle_speed" in tdf.columns and tdf["vehicle_speed"].notna().any():
            result["trip_mean_vehicle_speed"] = tdf["vehicle_speed"].mean()
            result["trip_max_vehicle_speed"] = tdf["vehicle_speed"].max()
        else:
            result["trip_mean_vehicle_speed"] = np.nan
            result["trip_max_vehicle_speed"] = np.nan

        # Wh/km for reference
        if distance > 0:
            result["trip_wh_per_km"] = result["trip_energy_wh"] / (distance / 1000.0)
        else:
            result["trip_wh_per_km"] = np.nan

        return pd.Series(result)

    print("  Computing trip aggregates ...")
    summaries = df.groupby("trip_id").apply(agg_trip).reset_index()
    return summaries


def main():
    in_path = os.path.join(cfg.DATA_DIR, "features.parquet")
    print(f"Reading {in_path} ...")
    df = pd.read_parquet(in_path)
    print(f"Loaded: {len(df):,} rows × {len(df.columns)} columns\n")

    # --- Segment ---
    print("Segmenting into micro-trips ...")
    df = segment_trips(df)
    n_sessions = df["session_id"].nunique()
    n_trips = df["trip_id"].nunique()
    print(f"  Sessions: {n_sessions:,}")
    print(f"  Micro-trips (before filter): {n_trips:,}\n")

    # --- Trip summaries ---
    summaries = compute_trip_summaries(df)

    # --- Filter trips ---
    before = len(summaries)
    summaries = summaries[
        (summaries["trip_n_steps"] >= cfg.MIN_TRIP_STEPS) &
        (summaries["trip_distance_m"] >= cfg.MIN_TRIP_DIST_M) &
        (summaries["trip_energy_wh"] > 0)
    ].reset_index(drop=True)
    print(f"Trip filtering:")
    print(f"  n_steps < {cfg.MIN_TRIP_STEPS} or dist < {cfg.MIN_TRIP_DIST_M}m or energy <= 0: "
          f"{before - len(summaries):>6,} trips removed")
    print(f"  Remaining trips: {len(summaries):,}\n")

    # Keep only rows belonging to valid trips
    valid_trips = set(summaries["trip_id"])
    df = df[df["trip_id"].isin(valid_trips)].reset_index(drop=True)

    # --- Save ---
    trips_path = os.path.join(cfg.DATA_DIR, "trips.parquet")
    summaries_path = os.path.join(cfg.DATA_DIR, "trip_summaries.parquet")
    df.to_parquet(trips_path, index=False)
    summaries.to_parquet(summaries_path, index=False)

    print(f"Saved row-level:   {len(df):,} rows → {trips_path}")
    print(f"Saved summaries:   {len(summaries):,} trips → {summaries_path}")

    # Stats
    print(f"\nTrip statistics:")
    for col in ["trip_n_steps", "trip_duration_s", "trip_distance_m", "trip_energy_wh", "trip_wh_per_km"]:
        if col in summaries.columns:
            s = summaries[col].dropna()
            print(f"  {col:<30} mean={s.mean():8.1f}  median={s.median():8.1f}  "
                  f"min={s.min():8.1f}  max={s.max():8.1f}")


if __name__ == "__main__":
    main()
