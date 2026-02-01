"""
Building footprints integration.

Goal:
Use higher-quality building outlines (e.g. Microsoft/Google footprints) where OSM is too coarse,
while keeping OSM height/levels/roof tags when available.

We support a production-friendly format:
- a single GeoPackage (GPKG) in EPSG:4326 with a "buildings" layer
  (so we can read only a bbox window quickly).

Env:
- FOOTPRINTS_GPKG_PATH=cache/footprints/ukraine_buildings.gpkg
- FOOTPRINTS_LAYER=buildings
- USE_FOOTPRINTS=1
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import os

import geopandas as gpd
import numpy as np
from shapely.geometry import box


HEIGHT_KEYS: List[str] = [
    "height",
    "building:height",
    "building:levels",
    "building:levels:aboveground",
    "roof:height",
    "roof:levels",
    "roof:shape",
    "building",
    "name",
]


def _get_cfg_path() -> Optional[Path]:
    # 1) Explicit env var wins
    p = os.getenv("FOOTPRINTS_GPKG_PATH")
    if p:
        return Path(p)
    # 2) Auto-detect default cache path (so user doesn't have to configure)
    default = Path("cache/footprints/ukraine_buildings.gpkg")
    if default.exists():
        return default
    return None


def is_footprints_enabled() -> bool:
    if (os.getenv("USE_FOOTPRINTS") or "0").lower() in ("1", "true", "yes"):
        return True
    p = _get_cfg_path()
    return bool(p and p.exists())


def load_footprints_bbox(
    north: float,
    south: float,
    east: float,
    west: float,
    target_crs: Optional[object],
) -> gpd.GeoDataFrame:
    """
    Loads footprints polygons within bbox.
    Input bbox is lat/lon degrees.
    Output is projected to target_crs if provided.
    """
    gpkg = _get_cfg_path()
    if not gpkg or not gpkg.exists():
        return gpd.GeoDataFrame()

    layer = os.getenv("FOOTPRINTS_LAYER") or "buildings"
    bbox_geom = box(west, south, east, north)  # EPSG:4326

    try:
        # Prefer pyogrio engine (faster and avoids fiona API differences on some installs)
        gdf = gpd.read_file(gpkg, layer=layer, bbox=bbox_geom, engine="pyogrio")
    except Exception:
        # Some drivers don't support bbox filter; fall back to read all then clip
        gdf = gpd.read_file(gpkg, layer=layer, engine="pyogrio")
        if not gdf.empty:
            gdf = gdf[gdf.geometry.intersects(bbox_geom)]

    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame()

    gdf = gdf[gdf.geometry.notna()].copy()
    try:
        gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    except Exception:
        pass

    if target_crs is not None:
        try:
            gdf = gdf.to_crs(target_crs)
        except Exception:
            # if unknown, keep as-is
            pass

    gdf["__from_footprints"] = True
    return gdf


def transfer_osm_attributes_to_footprints(
    footprints: gpd.GeoDataFrame,
    osm_buildings: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Spatially transfers OSM attributes (height/levels/roof/building) onto footprints.
    We keep footprint geometry but copy best-matching OSM tags.
    """
    if footprints is None or footprints.empty:
        return footprints if footprints is not None else gpd.GeoDataFrame()
    if osm_buildings is None or osm_buildings.empty:
        return footprints

    # Split out building parts: don't use them for matching main outline heights
    osm_main = osm_buildings
    if "__is_building_part" in osm_buildings.columns:
        osm_main = osm_buildings[~osm_buildings["__is_building_part"].fillna(False)]

    if osm_main.empty:
        return footprints

    cols = [c for c in HEIGHT_KEYS if c in osm_main.columns]
    osm_sub = osm_main[cols + ["geometry"]].copy()

    # Use intersects join, then pick the best match by max intersection area
    try:
        joined = gpd.sjoin(
            footprints[["geometry"]].copy(),
            osm_sub,
            how="left",
            predicate="intersects",
        )
    except Exception:
        # If sjoin fails (no spatial index), fall back to centroid-based join
        fp = footprints.copy()
        fp_cent = fp.copy()
        fp_cent["geometry"] = fp_cent.geometry.centroid
        joined = gpd.sjoin(fp_cent[["geometry"]], osm_sub, how="left", predicate="within")

    if joined.empty:
        return footprints

    # Compute overlap score if possible
    try:
        fp_geom = footprints.loc[joined.index, "geometry"].values
        osm_geom = osm_sub.loc[joined["index_right"].values, "geometry"].values
        inter_area = np.array([a.intersection(b).area if (a is not None and b is not None) else 0.0 for a, b in zip(fp_geom, osm_geom)])
        joined = joined.assign(__score=inter_area)
    except Exception:
        joined = joined.assign(__score=0.0)

    # For each footprint, pick the row with max score (or first)
    joined = joined.reset_index().rename(columns={"index": "__fp_index"})
    joined_sorted = joined.sort_values(["__fp_index", "__score"], ascending=[True, False])
    best = joined_sorted.drop_duplicates(subset=["__fp_index"], keep="first")
    best = best.set_index("__fp_index")

    out = footprints.copy()
    for c in cols:
        if c in best.columns:
            out[c] = best[c].reindex(out.index).values

    return out


