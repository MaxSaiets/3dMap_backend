"""
Модуль для попереднього завантаження всіх OSM даних при старті сервера
та використання spatial index для швидкого вибору об'єктів за bbox.
"""
import os
import warnings
import time
from pathlib import Path
from typing import Tuple, Optional
import geopandas as gpd
import pandas as pd

warnings.filterwarnings('ignore', category=DeprecationWarning, module='pandas')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='geopandas')

# Глобальні змінні для preloaded даних
_preloaded_buildings: Optional[gpd.GeoDataFrame] = None
_preloaded_water: Optional[gpd.GeoDataFrame] = None
_preloaded_roads: Optional[gpd.GeoDataFrame] = None
_preloaded_green: Optional[gpd.GeoDataFrame] = None

_is_loaded = False


def load_all_data() -> None:
    """
    Завантажує всі дані з PBF файлу один раз при старті сервера.
    """
    global _preloaded_buildings, _preloaded_water, _preloaded_roads
    global _preloaded_green, _is_loaded
    
    if _is_loaded:
        print("[preload] Дані вже завантажені")
        return
    
    print("=" * 80)
    print("[preload] ПОЧАТОК ЗАВАНТАЖЕННЯ ВСІХ ДАНИХ З PBF")
    print("=" * 80)
    
    pbf_path = Path(os.getenv("OSM_PBF_PATH") or "cache/osm/ukraine-latest.osm.pbf")
    if not pbf_path.exists():
        print(f"[ERROR] PBF файл не знайдено: {pbf_path}")
        print("[preload] Переходимо на режим завантаження по запиту")
        return
    
    try:
        sz = pbf_path.stat().st_size
        print(f"[preload] PBF файл: {pbf_path} ({sz / (1024 * 1024):.1f} MiB)")
    except Exception:
        pass
    
    from pyrosm import OSM
    
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Завантажуємо БЕЗ bounding_box - всі дані для України
    t0 = time.perf_counter()
    print("[preload] Ініціалізація OSM (може зайняти 1-3 хвилини)...")
    osm = OSM(str(pbf_path))
    t1 = time.perf_counter()
    print(f"[preload] OSM ініціалізовано за {(t1 - t0):.2f}s")
    
    # 1. Будівлі
    print("[preload] Завантаження будівель...")
    buildings_extra = [
        "building", "name", "height", "building:height",
        "building:levels", "building:levels:aboveground",
        "roof:height", "roof:levels", "roof:shape",
    ]
    t_b_start = time.perf_counter()
    _preloaded_buildings = osm.get_buildings(extra_attributes=buildings_extra)
    _preloaded_buildings = _preloaded_buildings if _preloaded_buildings is not None else gpd.GeoDataFrame()
    
    # Опціонально: building:part
    fetch_parts = (os.getenv("OSM_FETCH_BUILDING_PARTS") or "0").lower() in ("1", "true", "yes")
    if fetch_parts:
        print("[preload] Завантаження building:part...")
        parts = osm.get_data_by_custom_criteria(
            custom_filter={"building:part": True},
            osm_keys_to_keep=[
                "building:part", "height", "building:height",
                "building:levels", "building:levels:aboveground",
                "roof:height", "roof:levels", "roof:shape", "name",
            ],
            filter_type="keep",
            keep_nodes=False,
            keep_relations=True,
            keep_ways=True,
        )
        parts = parts if parts is not None else gpd.GeoDataFrame()
        if not parts.empty:
            parts = parts.copy()
            parts["__is_building_part"] = True
            # Фільтр частин з висотою
            has_height = None
            for col in ["height", "building:height", "building:levels", 
                       "building:levels:aboveground", "roof:height", "roof:levels"]:
                if col in parts.columns:
                    s = parts[col].notna()
                    has_height = s if has_height is None else (has_height | s)
            if has_height is not None:
                parts = parts[has_height]
            
            if not parts.empty:
                if _preloaded_buildings.empty:
                    _preloaded_buildings = parts
                else:
                    _preloaded_buildings = gpd.GeoDataFrame(
                        pd.concat([_preloaded_buildings, parts], ignore_index=True),
                        crs=_preloaded_buildings.crs or parts.crs,
                    )
    
    _preloaded_buildings = _preloaded_buildings[_preloaded_buildings.geometry.notna()] if not _preloaded_buildings.empty else _preloaded_buildings
    t_b_end = time.perf_counter()
    print(f"[preload] Будівлі: {len(_preloaded_buildings)} за {(t_b_end - t_b_start):.2f}s")
    
    # 2. Вода
    print("[preload] Завантаження водних об'єктів...")
    t_w_start = time.perf_counter()
    _preloaded_water = osm.get_data_by_custom_criteria(
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
    _preloaded_water = _preloaded_water if _preloaded_water is not None else gpd.GeoDataFrame()
    _preloaded_water = _preloaded_water[_preloaded_water.geometry.notna()] if not _preloaded_water.empty else _preloaded_water
    t_w_end = time.perf_counter()
    print(f"[preload] Вода: {len(_preloaded_water)} за {(t_w_end - t_w_start):.2f}s")
    
    # 3. Дороги
    print("[preload] Завантаження доріг...")
    roads_nt = (os.getenv("OSM_ROADS_NETWORK_TYPE") or "drive").strip().lower()
    if roads_nt == "drive":
        roads_nt = "driving"
    elif roads_nt == "walk":
        roads_nt = "walking"
    elif roads_nt in ("bike", "bicycle"):
        roads_nt = "cycling"
    
    roads_extra = [
        "highway", "bridge", "bridge:type", "bridge:structure",
        "man_made", "tunnel", "layer", "name", "oneway",
        "lanes", "maxspeed", "surface",
    ]
    t_r_start = time.perf_counter()
    _preloaded_roads = osm.get_network(
        network_type=roads_nt,
        extra_attributes=roads_extra,
        nodes=False
    )
    _preloaded_roads = _preloaded_roads if _preloaded_roads is not None else gpd.GeoDataFrame()
    _preloaded_roads = _preloaded_roads[_preloaded_roads.geometry.notna()] if not _preloaded_roads.empty else _preloaded_roads
    t_r_end = time.perf_counter()
    print(f"[preload] Дороги: {len(_preloaded_roads)} за {(t_r_end - t_r_start):.2f}s")
    
    # 4. Зелені зони
    print("[preload] Завантаження зелених зон...")
    t_g_start = time.perf_counter()
    _preloaded_green = osm.get_data_by_custom_criteria(
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
    _preloaded_green = _preloaded_green if _preloaded_green is not None else gpd.GeoDataFrame()
    _preloaded_green = _preloaded_green[_preloaded_green.geometry.notna()] if not _preloaded_green.empty else _preloaded_green
    _preloaded_green = _preloaded_green[_preloaded_green.geom_type.isin(["Polygon", "MultiPolygon"])] if not _preloaded_green.empty else _preloaded_green
    t_g_end = time.perf_counter()
    print(f"[preload] Зелені зони: {len(_preloaded_green)} за {(t_g_end - t_g_start):.2f}s")
    
    print(f"  Зелені зони: {len(_preloaded_green)}")
    
    # Оцінка пам'яті
    
    # Оцінка пам'яті
    try:
        import sys
        total_mb = 0
        for gdf_name, gdf in [
            ("buildings", _preloaded_buildings),
            ("water", _preloaded_water),
            ("roads", _preloaded_roads),
            ("green", _preloaded_green),
        ]:
            if gdf is not None and not gdf.empty:
                size_mb = sys.getsizeof(gdf) / (1024 * 1024)
                total_mb += size_mb
                print(f"  {gdf_name}: ~{size_mb:.1f} MB")
        print(f"  Загалом: ~{total_mb:.1f} MB в пам'яті")
    except Exception:
        pass
    
    print("=" * 80)


def get_data_for_bbox(
    north: float,
    south: float,
    east: float,
    west: float
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Отримує дані для вказаного bbox з preloaded даних, використовуючи spatial index.
    
    Returns:
        (buildings_gdf, water_gdf, roads_gdf)
    """
    global _preloaded_buildings, _preloaded_water, _preloaded_roads, _is_loaded
    
    if not _is_loaded:
        # Якщо дані не завантажені, повертаємо порожні
        print("[WARN] Preloaded дані не завантажені, повертаємо порожні GeoDataFrames")
        return gpd.GeoDataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame()
    
    # Використовуємо spatial index напряму з WGS84 координатами
    # Spatial index працює з bounds (bbox) як tuple (minx, miny, maxx, maxy)
    # Але наші дані вже в UTM, тому потрібно конвертувати bbox в UTM
    
    from shapely.geometry import box
    import osmnx as ox
    
    # Визначаємо target CRS з наявних даних
    if not _preloaded_buildings.empty:
        target_crs = _preloaded_buildings.crs
    elif not _preloaded_water.empty:
        target_crs = _preloaded_water.crs
    elif not _preloaded_roads.empty:
        target_crs = _preloaded_roads.crs
    else:
        target_crs = None
    
    if target_crs is None:
        print("[WARN] Неможливо визначити CRS, повертаємо порожні GeoDataFrames")
        return gpd.GeoDataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame()
    
    # Створюємо bbox в WGS84 та конвертуємо в UTM
    bbox_wgs84 = box(west, south, east, north)
    bbox_gdf = gpd.GeoDataFrame([1], geometry=[bbox_wgs84], crs="EPSG:4326")
    bbox_gdf = bbox_gdf.to_crs(target_crs)
    bbox_geom = bbox_gdf.geometry.iloc[0]
    bbox_bounds = bbox_geom.bounds  # (minx, miny, maxx, maxy)
    
    # Використовуємо spatial index для швидкого вибору
    buildings_result = gpd.GeoDataFrame()
    water_result = gpd.GeoDataFrame()
    roads_result = gpd.GeoDataFrame()
    
    if not _preloaded_buildings.empty:
        try:
            # Використовуємо spatial index для швидкого вибору
            possible_matches_index = list(_preloaded_buildings.sindex.intersection(bbox_bounds))
            if possible_matches_index:
                buildings_result = _preloaded_buildings.iloc[possible_matches_index].copy()
                # Фільтруємо тільки ті, що дійсно перетинаються з bbox
                buildings_result = buildings_result[buildings_result.geometry.intersects(bbox_geom)]
        except Exception as e:
            print(f"[WARN] Помилка вибору будівель: {e}")
    
    if not _preloaded_water.empty:
        try:
            possible_matches_index = list(_preloaded_water.sindex.intersection(bbox_bounds))
            if possible_matches_index:
                water_result = _preloaded_water.iloc[possible_matches_index].copy()
                water_result = water_result[water_result.geometry.intersects(bbox_geom)]
        except Exception as e:
            print(f"[WARN] Помилка вибору води: {e}")
    
    if not _preloaded_roads.empty:
        try:
            possible_matches_index = list(_preloaded_roads.sindex.intersection(bbox_bounds))
            if possible_matches_index:
                roads_result = _preloaded_roads.iloc[possible_matches_index].copy()
                roads_result = roads_result[roads_result.geometry.intersects(bbox_geom)]
        except Exception as e:
            print(f"[WARN] Помилка вибору доріг: {e}")
    
    print(f"[preload] Вибрано для bbox: {len(buildings_result)} будівель, {len(water_result)} вод, {len(roads_result)} доріг")
    
    return buildings_result, water_result, roads_result


def get_extras_for_bbox(
    north: float,
    south: float,
    east: float,
    west: float
) -> gpd.GeoDataFrame:
    """
    Отримує extras (green) для вказаного bbox.
    
    Returns:
        green_gdf
    """
    global _preloaded_green, _is_loaded
    
    if not _is_loaded:
        return gpd.GeoDataFrame()
    
    from shapely.geometry import box
    
    bbox_wgs84 = box(west, south, east, north)
    
    if not _preloaded_green.empty:
        target_crs = _preloaded_green.crs
    else:
        target_crs = None
    
    if target_crs is None:
        return gpd.GeoDataFrame()
    
    bbox_gdf = gpd.GeoDataFrame([1], geometry=[bbox_wgs84], crs="EPSG:4326")
    bbox_gdf = bbox_gdf.to_crs(target_crs)
    bbox_geom = bbox_gdf.geometry.iloc[0]
    bbox_bounds = bbox_geom.bounds
    
    green_result = gpd.GeoDataFrame()
    
    if not _preloaded_green.empty:
        try:
            possible_matches_index = list(_preloaded_green.sindex.intersection(bbox_bounds))
            if possible_matches_index:
                green_result = _preloaded_green.iloc[possible_matches_index].copy()
                green_result = green_result[green_result.geometry.intersects(bbox_geom)]
        except Exception as e:
            print(f"[WARN] Помилка вибору зелених зон: {e}")
    
    return green_result


def is_loaded() -> bool:
    """Перевіряє, чи завантажені дані"""
    return _is_loaded

