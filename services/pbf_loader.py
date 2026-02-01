"""
Local OSM PBF loader (Geofabrik) for Ukraine-wide best data.

Goal:
- Avoid Overpass instability/rate limits
- Enable reliable, repeatable results for production

Uses pyrosm to extract features by bbox from a local .osm.pbf.
Optionally auto-downloads the Ukraine PBF from Geofabrik.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional
import os
import warnings
import hashlib
import time
import gc

import numpy as np

import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests


GEOFABRIK_UKRAINE_PBF_URL = "https://download.geofabrik.de/europe/ukraine-latest.osm.pbf"


def _pyrosm_network_type() -> str:
    """
    Map our env-style OSM_ROADS_NETWORK_TYPE (osmnx-like) to pyrosm names.
    """
    nt = (os.getenv("OSM_ROADS_NETWORK_TYPE") or "drive").strip().lower()
    mapping = {
        "drive": "driving",
        "driving": "driving",
        "walk": "walking",
        "walking": "walking",
        "bike": "cycling",
        "bicycle": "cycling",
        "cycling": "cycling",
        "all": "all",
    }
    return mapping.get(nt, nt)


def _pbf_cache_dir() -> Path:
    # Disk cache for extracted bbox datasets (Parquet). Greatly reduces repeated PBF scans.
    return Path(os.getenv("OSM_PBF_CACHE_DIR") or "cache/osm/pbf_bbox_cache")


def _cache_enabled() -> bool:
    # Кеш вимкнено за замовчуванням для меншого використання пам'яті та дискового простору
    return (os.getenv("OSM_PBF_DISK_CACHE") or "0").lower() in ("1", "true", "yes")


def _bbox_cache_key(
    kind: str,
    north: float,
    south: float,
    east: float,
    west: float,
    *params: object,
) -> str:
    # Round to avoid cache fragmentation due to tiny float diffs
    s = f"{kind}|{round(float(north), 6)}|{round(float(south), 6)}|{round(float(east), 6)}|{round(float(west), 6)}|"
    s += "|".join([str(p) for p in params])
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _read_gdf_parquet(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(path)
    # Ensure it's a GeoDataFrame
    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf)
    return gdf


def _write_gdf_parquet(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # keep index off for stable cache size
    gdf.to_parquet(path, index=False)


def _minimal_columns(gdf: gpd.GeoDataFrame, keep: list[str]) -> gpd.GeoDataFrame:
    if gdf is None or getattr(gdf, "empty", True):
        return gdf
    cols = [c for c in keep if c in gdf.columns]
    if "geometry" not in cols and "geometry" in gdf.columns:
        cols = cols + ["geometry"]
    try:
        return gdf[cols].copy()
    except Exception:
        return gdf


def _download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(dst)


def ensure_ukraine_pbf(pbf_path: Path) -> Path:
    """
    Ensures the PBF exists. If not, optionally auto-downloads from Geofabrik.
    """
    if pbf_path.exists():
        return pbf_path
    auto = (os.getenv("OSM_PBF_AUTO_DOWNLOAD") or "1").lower() in ("1", "true", "yes")
    if not auto:
        raise FileNotFoundError(f"OSM PBF not found: {pbf_path}")
    print(f"[pbf] Downloading Ukraine PBF from Geofabrik to: {pbf_path}")
    _download_file(GEOFABRIK_UKRAINE_PBF_URL, pbf_path)
    return pbf_path


def fetch_city_data_from_pbf(
    north: float,
    south: float,
    east: float,
    west: float,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Returns (buildings_gdf, water_gdf, roads_edges_gdf) in projected CRS (UTM).
    """
    from pyrosm import OSM

    bbox = (north, south, east, west)
    # pyrosm expects [west, south, east, north]
    bb = [west, south, east, north]

    pbf_path = Path(os.getenv("OSM_PBF_PATH") or "cache/osm/ukraine-latest.osm.pbf")
    pbf_path = ensure_ukraine_pbf(pbf_path)
    try:
        sz = pbf_path.stat().st_size
        print(f"[pbf] Using PBF: {pbf_path} ({sz / (1024 * 1024):.1f} MiB)")
    except Exception:
        pass

    # Reduce noisy warnings from pyrosm/pandas
    warnings.filterwarnings("ignore", category=UserWarning)

    fetch_parts = (os.getenv("OSM_FETCH_BUILDING_PARTS") or "0").lower() in ("1", "true", "yes")
    roads_nt = _pyrosm_network_type()

    # Disk cache lookup (avoids rescanning 800MB PBF repeatedly)
    if _cache_enabled():
        key = _bbox_cache_key("city", north, south, east, west, roads_nt, int(fetch_parts))
        base = _pbf_cache_dir() / key
        bpath = base / "buildings.parquet"
        wpath = base / "water.parquet"
        rpath = base / "roads_edges.parquet"
        if bpath.exists() and wpath.exists() and rpath.exists():
            try:
                buildings = _read_gdf_parquet(bpath)
                water = _read_gdf_parquet(wpath)
                roads = _read_gdf_parquet(rpath)
                print(f"[pbf] Disk cache hit: {base}")
                return buildings, water, roads
            except Exception as e:
                print(f"[pbf] Disk cache read failed, rebuilding: {e}")

    t0 = time.perf_counter()
    # MEMORY OPTIMIZATION: OSM bounding_box scan can be memory-intensive
    # Create OSM object with bbox to limit data loaded into memory
    osm = OSM(str(pbf_path), bounding_box=bb)
    t1 = time.perf_counter()
    print(f"[pbf] OSM(bbox) init: {(t1 - t0):.2f}s (this may scan the big PBF)")
    gc.collect()  # Free any temporary memory from OSM init

    # Buildings
    buildings_extra = [
        "building",
        "name",
        "height",
        "building:height",
        "building:levels",
        "building:levels:aboveground",
        "roof:height",
        "roof:levels",
        "roof:shape",
    ]
    buildings = osm.get_buildings(extra_attributes=buildings_extra)
    buildings = buildings if buildings is not None else gpd.GeoDataFrame()
    buildings = _minimal_columns(buildings, buildings_extra + ["geometry"])
    t2 = time.perf_counter()
    print(f"[pbf] buildings: {len(buildings)} in {(t2 - t1):.2f}s")

    # building:part is optional (can be expensive on dense areas)
    parts = gpd.GeoDataFrame()
    if fetch_parts:
        parts = osm.get_data_by_custom_criteria(
            custom_filter={"building:part": True},
            osm_keys_to_keep=[
                "building:part",
                "height",
                "building:height",
                "building:levels",
                "building:levels:aboveground",
                "roof:height",
                "roof:levels",
                "roof:shape",
                "name",
            ],
            filter_type="keep",
            keep_nodes=False,
            keep_relations=True,
            keep_ways=True,
        )
        parts = parts if parts is not None else gpd.GeoDataFrame()
    t3 = time.perf_counter()
    if fetch_parts:
        print(f"[pbf] building:part: {len(parts)} in {(t3 - t2):.2f}s")

    if not parts.empty:
        parts = parts.copy()
        parts["__is_building_part"] = True
        # Keep only parts that carry height/levels/roof info to avoid duplicates
        has_height = None
        for col in [
            "height",
            "building:height",
            "building:levels",
            "building:levels:aboveground",
            "roof:height",
            "roof:levels",
        ]:
            if col in parts.columns:
                s = parts[col].notna()
                has_height = s if has_height is None else (has_height | s)
        if has_height is not None:
            parts = parts[has_height]

    buildings = buildings[buildings.geometry.notna()] if not buildings.empty else buildings
    parts = parts[parts.geometry.notna()] if not parts.empty else parts

    # Water polygons
    water = osm.get_data_by_custom_criteria(
        custom_filter={
            "natural": ["water"],
            "waterway": ["riverbank"],
            "landuse": ["reservoir"],
            "water": True,
        },
        filter_type="keep",
        keep_nodes=False,
        keep_relations=True,
        keep_ways=True,
    )
    water = water if water is not None else gpd.GeoDataFrame()
    water = water[water.geometry.notna()] if not water.empty else water
    t4 = time.perf_counter()
    print(f"[pbf] water: {len(water)} in {(t4 - t3):.2f}s")

    # Roads as edges GeoDataFrame
    roads_extra = [
        "highway",
        "bridge",
        "bridge:type",
        "bridge:structure",
        "man_made",
        "tunnel",
        "layer",
        "name",
        "oneway",
        "lanes",
        "maxspeed",
        "surface",
    ]
    roads = osm.get_network(network_type=roads_nt, extra_attributes=roads_extra, nodes=False)
    roads = roads if roads is not None else gpd.GeoDataFrame()
    roads = roads[roads.geometry.notna()] if not roads.empty else roads
    roads = _minimal_columns(roads, roads_extra + ["geometry"])
    t5 = time.perf_counter()
    print(f"[pbf] roads edges: {len(roads)} in {(t5 - t4):.2f}s (network_type={roads_nt})")

    # Merge buildings + parts
    if not parts.empty:
        if buildings.empty:
            buildings = parts
        else:
            buildings = gpd.GeoDataFrame(
                pd.concat([buildings, parts], ignore_index=True),
                crs=buildings.crs or parts.crs,
            )

    # Project all to UTM (consistent with current pipeline)
    if not buildings.empty:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            buildings = ox.project_gdf(buildings)
    if not water.empty:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            water = ox.project_gdf(water)
    if not roads.empty:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            roads = ox.project_gdf(roads)
    t6 = time.perf_counter()
    print(f"[pbf] project_gdf (all): {(t6 - t5):.2f}s")

    # Free pyrosm internals as early as possible (helps RAM on Windows)
    try:
        del osm
    except Exception:
        pass
    gc.collect()

    # Sanity stats (lightweight, helps confirm that "required data exists" in the PBF)
    try:
        hw = 0
        if roads is not None and not roads.empty and "highway" in roads.columns:
            hw = int(np.sum(roads["highway"].notna()))
        print(f"[pbf] Loaded: {len(buildings)} buildings, {len(water)} water, {len(roads)} roads edges (highway tags: {hw})")
    except Exception:
        print(f"[pbf] Loaded: {len(buildings)} buildings, {len(water)} water, {len(roads)} roads edges from PBF")

    # Disk cache write
    if _cache_enabled():
        try:
            key = _bbox_cache_key("city", north, south, east, west, roads_nt, int(fetch_parts))
            base = _pbf_cache_dir() / key
            _write_gdf_parquet(buildings, base / "buildings.parquet")
            _write_gdf_parquet(water, base / "water.parquet")
            _write_gdf_parquet(roads, base / "roads_edges.parquet")
            print(f"[pbf] Disk cache saved: {base}")
        except Exception as e:
            print(f"[pbf] Disk cache write failed: {e}")
    return buildings, water, roads


def fetch_extras_from_pbf(
    north: float,
    south: float,
    east: float,
    west: float,
) -> gpd.GeoDataFrame:
    """
    Returns (green_polygons_gdf, poi_points_gdf) in projected CRS (UTM).
    """
    from pyrosm import OSM

    bb = [west, south, east, north]
    pbf_path = Path(os.getenv("OSM_PBF_PATH") or "cache/osm/ukraine-latest.osm.pbf")
    pbf_path = ensure_ukraine_pbf(pbf_path)

    # Disk cache lookup
    if _cache_enabled():
        key = _bbox_cache_key("extras", north, south, east, west)
        base = _pbf_cache_dir() / key
        gpath = base / "green.parquet"
        if gpath.exists():
            try:
                green = _read_gdf_parquet(gpath)
                print(f"[pbf] Disk cache hit (extras): {base}")
                return green
            except Exception as e:
                print(f"[pbf] Disk cache read failed (extras), rebuilding: {e}")

    warnings.filterwarnings("ignore", category=UserWarning)
    t0 = time.perf_counter()
    osm = OSM(str(pbf_path), bounding_box=bb)
    t1 = time.perf_counter()
    print(f"[pbf] OSM(bbox) init (extras): {(t1 - t0):.2f}s")

    green = osm.get_data_by_custom_criteria(
        custom_filter={
            "leisure": ["park", "garden", "playground", "recreation_ground", "pitch"],
            "landuse": ["grass", "meadow", "forest", "village_green"],
            "natural": ["wood"],
        },
        filter_type="keep",
        keep_nodes=False,
        keep_relations=True,
        keep_ways=True,
    )
    green = green if green is not None else gpd.GeoDataFrame()
    green = green[green.geometry.notna()] if not green.empty else green
    if not green.empty:
        green = green[green.geom_type.isin(["Polygon", "MultiPolygon"])]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            green = ox.project_gdf(green)
    t2 = time.perf_counter()
    print(f"[pbf] green: {len(green)} in {(t2 - t1):.2f}s")

    try:
        del osm
    except Exception:
        pass
    gc.collect()

    print(f"[pbf] Extras: {len(green)} green polygons")

    if _cache_enabled():
        try:
            key = _bbox_cache_key("extras", north, south, east, west)
            base = _pbf_cache_dir() / key
            _write_gdf_parquet(green, base / "green.parquet")
            print(f"[pbf] Disk cache saved (extras): {base}")
        except Exception as e:
            print(f"[pbf] Disk cache write failed (extras): {e}")
    return green


