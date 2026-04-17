"""
01_ingest.py

Reads all raw CSVs from ../all_data, keeps only the columns needed for the
pipeline, maps each unique IMEI to a stable integer rider_id, sorts by
rider + timestamp, and writes a single Parquet file.

Output: data/raw_ingested.parquet
"""

import glob
import os

import pandas as pd

from importlib import import_module
cfg = import_module("00_config")


def main():
    os.makedirs(cfg.DATA_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(cfg.RAW_DIR, "*.csv")))
    # Filter out macOS metadata files
    files = [f for f in files if not os.path.basename(f).startswith("._")]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {cfg.RAW_DIR}")

    print(f"Found {len(files)} input files. Reading...")

    chunks = []
    for i, path in enumerate(files, 1):
        df = pd.read_csv(path, low_memory=False)
        # Keep only columns that exist in this file
        keep = [c for c in cfg.INGEST_COLS if c in df.columns]
        df = df[keep]
        chunks.append(df)
        print(f"  [{i:02d}/{len(files)}] {os.path.basename(path)}: {len(df):>8,} rows")

    data = pd.concat(chunks, ignore_index=True)
    print(f"\nTotal rows loaded: {len(data):,}")

    # --- Assign stable rider_id from IMEI ordering ---
    imei_order = data["imei_no"].drop_duplicates().reset_index(drop=True)
    imei_to_rider = {imei: idx + 1 for idx, imei in enumerate(imei_order)}
    data["rider_id"] = data["imei_no"].map(imei_to_rider)
    data = data.drop(columns=["imei_no"])

    print(f"Unique riders (IMEIs): {len(imei_to_rider)}")

    # --- Parse timestamps and sort ---
    data["gps_date"] = pd.to_datetime(data["gps_date"], utc=True, errors="coerce")
    data = data.dropna(subset=["gps_date"])
    data = data.sort_values(["rider_id", "gps_date"]).reset_index(drop=True)

    # --- Save ---
    out_path = os.path.join(cfg.DATA_DIR, "raw_ingested.parquet")
    data.to_parquet(out_path, index=False)
    print(f"\nSaved: {len(data):,} rows × {len(data.columns)} columns → {out_path}")

    # Quick column summary
    print(f"\nColumns ({len(data.columns)}):")
    for col in data.columns:
        non_null = data[col].notna().mean() * 100
        print(f"  {col:<25} {non_null:5.1f}% non-null")


if __name__ == "__main__":
    main()
