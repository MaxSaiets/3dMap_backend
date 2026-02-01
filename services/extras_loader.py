"""
Extra layers loader:
- parks/green areas (polygons)
- POIs (benches etc.)

Works in two modes:
- OSM_SOURCE=pbf -> read from local Geofabrik PBF via pyrosm
- otherwise -> fetch from Overpass via OSMnx (best-effort)
"""

from __future__ import annotations

import os
import warnings
from typing import Tuple

import geopandas as gpd
import osmnx as ox


# Кеш ВИМКНЕНО: завжди завантажуємо свіжі дані для кожної зони
# _EXTRAS_CACHE: dict[tuple, tuple[float, gpd.GeoDataFrame, gpd.GeoDataFrame]] = {}  # DISABLED


def _bbox_key(north: float, south: float, east: float, west: float) -> tuple[float, float, float, float]:
    return (round(float(north), 6), round(float(south), 6), round(float(east), 6), round(float(west), 6))


def fetch_extras(
    north: float,
    south: float,
    east: float,
    west: float,
) -> gpd.GeoDataFrame:
    # Перевіряємо чи є preloaded дані (пріоритет)
    try:
        from services.preloaded_data import is_loaded, get_extras_for_bbox
        if is_loaded():
            print("[extras_loader] Використовую preloaded дані")
            green, pois = get_extras_for_bbox(north, south, east, west)
            return green, pois
    except Exception as e:
        print(f"[WARN] Помилка використання preloaded даних для extras: {e}, використовуємо звичайний режим")
    
    # Використовуємо Overpass API за замовчуванням (завантажує тільки для конкретної зони, без кешу)
    # Для використання PBF встановіть OSM_SOURCE=pbf в .env
    source = (os.getenv("OSM_SOURCE") or "overpass").lower()

    # Кеш ВИМКНЕНО: завжди завантажуємо свіжі дані
    ttl_s = 0.0
    # k = (source, _bbox_key(north, south, east, west))  # DISABLED
    # import time as _time
    # now = _time.time()
    # if ttl_s > 0 and k in _EXTRAS_CACHE:  # DISABLED
    #     ...

    if source in ("pbf", "geofabrik", "local"):
        from services.pbf_loader import fetch_extras_from_pbf

        green, pois = fetch_extras_from_pbf(north, south, east, west)
        # Кеш вимкнено: не зберігаємо результати
        # if ttl_s > 0:
        #     _EXTRAS_CACHE[k] = (now, green, pois)  # DISABLED
        return green, pois

    bbox = (north, south, east, west)

    # Вимкнення кешу OSMnx для меншого використання пам'яті
    ox.settings.use_cache = False
    ox.settings.log_console = False

    # Parks/green polygons
    tags_green = {
        "leisure": ["park", "garden", "playground", "recreation_ground", "pitch"],
        "landuse": ["grass", "meadow", "forest", "village_green"],
        "natural": ["wood"],
    }
    # POIs - REMOVED per user request
    # tags_pois = { ... }

    gdf_green = gpd.GeoDataFrame()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            gdf_green = ox.features_from_bbox(*bbox, tags=tags_green)
        if not gdf_green.empty:
            gdf_green = gdf_green[gdf_green.geometry.notna()]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                gdf_green = ox.project_gdf(gdf_green)
            # Keep polygons only
            gdf_green = gdf_green[gdf_green.geom_type.isin(["Polygon", "MultiPolygon"])]
    except Exception:
        gdf_green = gpd.GeoDataFrame()

    return gdf_green


