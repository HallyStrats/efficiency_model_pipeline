"""
05_split.py

Splits data into train/val/test sets by rider. Same split used for both
GPS-only and full-feature models to ensure fair comparison.

Split strategy:
  - 80% riders for training, 20% for testing (stratified by total distance)
  - Within training riders, 10% of trips held out for validation

Output:
  data/train_trips.parquet, data/val_trips.parquet, data/test_trips.parquet
  data/train_summaries.parquet, data/val_summaries.parquet, data/test_summaries.parquet
  data/split_assignments.json
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from importlib import import_module
cfg = import_module("00_config")


def main():
    trips_path = os.path.join(cfg.DATA_DIR, "trips.parquet")
    summaries_path = os.path.join(cfg.DATA_DIR, "trip_summaries.parquet")

    print(f"Reading {trips_path} ...")
    trips = pd.read_parquet(trips_path)
    summaries = pd.read_parquet(summaries_path)
    print(f"  Rows: {len(trips):,}   Trips: {len(summaries):,}   Riders: {summaries['rider_id'].nunique()}\n")

    # --- Rider-level split ---
    rider_stats = summaries.groupby("rider_id").agg(
        total_distance=("trip_distance_m", "sum"),
        n_trips=("trip_id", "count"),
    ).reset_index()

    # Stratify by distance quartile
    rider_stats["dist_bin"] = pd.qcut(
        rider_stats["total_distance"], q=4, labels=False, duplicates="drop"
    )

    train_riders, test_riders = train_test_split(
        rider_stats["rider_id"],
        test_size=cfg.TEST_RIDER_FRAC,
        random_state=cfg.SEED,
        stratify=rider_stats["dist_bin"],
    )

    train_riders = set(train_riders)
    test_riders = set(test_riders)

    print(f"Rider split:")
    print(f"  Train riders: {len(train_riders)}")
    print(f"  Test riders:  {len(test_riders)}")

    # --- Within training riders, hold out validation trips ---
    train_summaries = summaries[summaries["rider_id"].isin(train_riders)].copy()
    train_trip_ids, val_trip_ids = train_test_split(
        train_summaries["trip_id"],
        test_size=cfg.VAL_TRIP_FRAC,
        random_state=cfg.SEED,
    )
    train_trip_ids = set(train_trip_ids)
    val_trip_ids = set(val_trip_ids)
    test_trip_ids = set(summaries[summaries["rider_id"].isin(test_riders)]["trip_id"])

    # --- Split row-level data ---
    train_trips = trips[trips["trip_id"].isin(train_trip_ids)].reset_index(drop=True)
    val_trips = trips[trips["trip_id"].isin(val_trip_ids)].reset_index(drop=True)
    test_trips = trips[trips["trip_id"].isin(test_trip_ids)].reset_index(drop=True)

    train_sum = summaries[summaries["trip_id"].isin(train_trip_ids)].reset_index(drop=True)
    val_sum = summaries[summaries["trip_id"].isin(val_trip_ids)].reset_index(drop=True)
    test_sum = summaries[summaries["trip_id"].isin(test_trip_ids)].reset_index(drop=True)

    print(f"\nSplit results:")
    print(f"  Train: {len(train_trips):>8,} rows, {len(train_sum):>6,} trips, {train_sum['rider_id'].nunique():>3} riders")
    print(f"  Val:   {len(val_trips):>8,} rows, {len(val_sum):>6,} trips, {val_sum['rider_id'].nunique():>3} riders")
    print(f"  Test:  {len(test_trips):>8,} rows, {len(test_sum):>6,} trips, {test_sum['rider_id'].nunique():>3} riders")

    # --- Save ---
    for name, data in [
        ("train_trips", train_trips), ("val_trips", val_trips), ("test_trips", test_trips),
        ("train_summaries", train_sum), ("val_summaries", val_sum), ("test_summaries", test_sum),
    ]:
        path = os.path.join(cfg.DATA_DIR, f"{name}.parquet")
        data.to_parquet(path, index=False)
        print(f"  Saved {path}")

    # Save split assignments
    assignments = {
        "train_riders": sorted(train_riders),
        "test_riders": sorted(test_riders),
        "train_trip_count": len(train_trip_ids),
        "val_trip_count": len(val_trip_ids),
        "test_trip_count": len(test_trip_ids),
    }
    split_path = os.path.join(cfg.DATA_DIR, "split_assignments.json")
    with open(split_path, "w") as f:
        json.dump(assignments, f, indent=2, default=int)
    print(f"  Saved {split_path}")

    # Quick energy distribution check
    print(f"\nEnergy distribution by split (Wh per trip):")
    for name, s in [("Train", train_sum), ("Val", val_sum), ("Test", test_sum)]:
        e = s["trip_energy_wh"]
        print(f"  {name:<6} mean={e.mean():.2f}  median={e.median():.2f}  "
              f"std={e.std():.2f}  min={e.min():.2f}  max={e.max():.2f}")


if __name__ == "__main__":
    main()
