"""
06b_osm_cache.py

Downloads the Nairobi road network from OpenStreetMap and builds a coordinate
→ highway_type lookup cache for fast road type feature engineering.

Uses osmnx to download the drivable road network, extracts (lat, long) →
highway_type mappings at 4 decimal places (~10m resolution), and saves
as a pickle file.

Output:
  data/osm_road_cache.pkl
"""

import os
import pickle

import numpy as np
import osmnx as ox
from scipy.spatial import cKDTree

from importlib import import_module
cfg = import_module("00_config")


def build_osm_cache():
    cache_path = os.path.join(cfg.DATA_DIR, "osm_road_cache.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"OSM cache already exists: {len(cache):,} entries → {cache_path}")
        return cache

    print("Downloading Nairobi road network from OSM...")
    G = ox.graph_from_place("Nairobi, Kenya", network_type="drive")
    edges = ox.graph_to_gdfs(G, nodes=False)
    print(f"  Edges: {len(edges):,}")

    # Extract reference points from edge geometries
    ref_points = []  # (lat, lon, highway_type)
    for _, row in edges.iterrows():
        hw = row.get("highway", "unknown")
        if isinstance(hw, list):
            hw = hw[0]
        coords = list(row.geometry.coords)
        for lon, lat in coords:
            ref_points.append((round(lat, 4), round(lon, 4), hw))

    print(f"  Reference points: {len(ref_points):,}")

    # Build KDTree for nearest-neighbour lookup
    ref_coords = np.array([(p[0], p[1]) for p in ref_points])
    ref_types = [p[2] for p in ref_points]
    tree = cKDTree(ref_coords)

    # Build cache: unique 4dp coordinates → nearest road type
    unique_coords = list(set((p[0], p[1]) for p in ref_points))

    # Also load the trip data to add all coordinates riders actually visit
    import pandas as pd
    for split in ["train_trips", "val_trips", "test_trips"]:
        path = os.path.join(cfg.DATA_DIR, f"{split}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path, columns=["lat", "long"])
            trip_coords = list(set(zip(
                np.round(df["lat"].values, 4),
                np.round(df["long"].values, 4),
            )))
            unique_coords.extend(trip_coords)

    unique_coords = list(set(unique_coords))
    print(f"  Unique coordinates to cache: {len(unique_coords):,}")

    # Query KDTree
    query_arr = np.array(unique_coords)
    dists, idxs = tree.query(query_arr)

    MAX_DIST_DEG = 0.001  # ~100m
    cache = {}
    for i, (lat, lon) in enumerate(unique_coords):
        if dists[i] < MAX_DIST_DEG:
            cache[(lat, lon)] = ref_types[idxs[i]]
        else:
            cache[(lat, lon)] = "unknown"

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    matched = sum(1 for v in cache.values() if v != "unknown")
    print(f"  Matched: {matched:,}/{len(cache):,} ({matched/len(cache)*100:.1f}%)")
    print(f"  Saved → {cache_path}")
    return cache


def main():
    build_osm_cache()


if __name__ == "__main__":
    main()
