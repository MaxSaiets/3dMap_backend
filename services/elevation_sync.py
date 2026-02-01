"""
Сервіс для синхронізації висот між зонами гексагональної сітки.
Забезпечує, що всі зони використовують один глобальний elevation_ref_m та baseline_offset_m.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from services.elevation_api import get_elevation_abs_meters_from_api
from services.crs_utils import bbox_latlon_to_utm
import os
import math


def _terrarium_min_elevation_in_bbox(
    bbox_latlon: Tuple[float, float, float, float],
    terrarium_zoom: Optional[int] = None,
) -> Optional[float]:
    """
    Robust global min for Terrarium: scan *all* tiles intersecting bbox and take min pixel elevation.
    This avoids missing minima due to sparse point sampling (which causes negative Z and seams).
    """
    provider = (os.getenv("ELEVATION_PROVIDER") or "terrarium").lower()
    if provider != "terrarium":
        return None
    try:
        from services.terrarium_tiles import TerrariumTileProvider, TileKey
    except Exception:
        return None

    north, south, east, west = bbox_latlon
    z = int(terrarium_zoom if terrarium_zoom is not None else os.getenv("TERRARIUM_ZOOM", "14"))
    base_url = os.getenv("TERRARIUM_URL", "https://s3.amazonaws.com/elevation-tiles-prod/terrarium")
    cache_dir = os.getenv("TERRARIUM_CACHE_DIR", "cache/terrarium")
    timeout = float(os.getenv("TERRARIUM_TIMEOUT", "30"))
    tp = TerrariumTileProvider(base_url=base_url, cache_dir=cache_dir, timeout=timeout)

    def latlon_to_tile_xy(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
        lat = max(min(lat, 85.05112878), -85.05112878)
        n = 2 ** zoom
        xtile = int((lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        ytile = int((1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
        return xtile, ytile

    # bbox corners -> tile range
    x1, y1 = latlon_to_tile_xy(west, north, z)
    x2, y2 = latlon_to_tile_xy(east, south, z)
    minx, maxx = (min(x1, x2), max(x1, x2))
    miny, maxy = (min(y1, y2), max(y1, y2))

    # IMPORTANT:
    # Terrarium tiles sometimes contain a handful of bogus extreme low pixels (e.g. ~-1800m) even in flat areas.
    # Using the absolute min pixel makes elevation_ref_m wildly wrong and produces "tower" bases (hundreds of mm).
    # We use a very low quantile per tile to be robust against such outliers.
    elev_min = None
    for tx in range(minx, maxx + 1):
        for ty in range(miny, maxy + 1):
            tile = tp.get_tile(TileKey(z=z, x=tx, y=ty))
            if tile is None:
                continue
            try:
                vals = np.asarray(tile, dtype=float).reshape(-1)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                # Drop impossible extremes / nodata-like spikes
                vals = vals[(vals > -5000.0) & (vals < 9000.0)]
                if vals.size == 0:
                    continue
                # 0.1% quantile ~ 65 pixels of a 256x256 tile; robust to a few bad pixels.
                tmin = float(np.quantile(vals, 0.001))
            except Exception:
                continue
            if not np.isfinite(tmin):
                continue
            elev_min = tmin if elev_min is None else min(elev_min, tmin)
    return elev_min


def calculate_global_elevation_reference(
    zones: List[Dict],
    source_crs: Optional[object] = None,
    terrarium_zoom: Optional[int] = None,
    z_scale: float = 1.0,
    sample_points_per_zone: int = 25,  # Кількість точок для семплінгу в кожній зоні
    global_center: Optional[object] = None,  # Глобальний центр для конвертації координат
    explicit_bbox: Optional[Tuple[float, float, float, float]] = None,  # Явний bbox (north, south, east, west)
) -> Tuple[Optional[float], float]:
    """
    Обчислює глобальний elevation_ref_m для всієї сітки зон.
    Це забезпечує, що всі зони використовують одну базову висоту для нормалізації.
    
    Args:
        zones: Список зон (GeoJSON features) з полями 'geometry'
        source_crs: CRS для перетворення координат (якщо None - визначається автоматично)
        terrarium_zoom: Zoom рівень для Terrarium tiles
        sample_points_per_zone: Кількість точок для семплінгу висот в кожній зоні
        explicit_bbox: Явний bbox для стабільності (north, south, east, west)
    
    Returns:
        Tuple (elevation_ref_m, baseline_offset_m):
        - elevation_ref_m: Глобальна базова висота (метри над рівнем моря) або None
        - baseline_offset_m: Зміщення baseline (метри) для мінімальної висоти на моделі
    """
    if not zones and explicit_bbox is None:
        return None, 0.0
    
    print(f"[INFO] Обчислення глобального elevation_ref для {len(zones) if zones else 0} зон...")
    
    grid_bbox_latlon = None
    
    if explicit_bbox is not None:
        grid_bbox_latlon = explicit_bbox
        print(f"[INFO] Використання явного grid bbox: north={grid_bbox_latlon[0]:.6f}, south={grid_bbox_latlon[1]:.6f}, "
              f"east={grid_bbox_latlon[2]:.6f}, west={grid_bbox_latlon[3]:.6f}")
    else:
        # Збираємо всі координати з усіх зон для визначення bbox
        all_lons = []
        all_lats = []
        
        for zone in zones:
            geometry = zone.get('geometry', {})
            if geometry.get('type') != 'Polygon':
                continue
            
            coordinates = geometry.get('coordinates', [])
            if not coordinates or len(coordinates) == 0:
                continue
            
            all_coords = [coord for ring in coordinates for coord in ring]
            zone_lons = [coord[0] for coord in all_coords]
            zone_lats = [coord[1] for coord in all_coords]
            
            all_lons.extend(zone_lons)
            all_lats.extend(zone_lats)
        
        if len(all_lons) == 0 or len(all_lats) == 0:
            print("[WARN] Не вдалося отримати координати зон, використовується локальна нормалізація")
            return None, 0.0
        
        # Обчислюємо bbox для всієї сітки
        grid_bbox_latlon = (
            max(all_lats),  # north
            min(all_lats),  # south
            max(all_lons),  # east
            min(all_lons)   # west
        )
        
        print(f"[INFO] Grid bbox (auto): north={grid_bbox_latlon[0]:.6f}, south={grid_bbox_latlon[1]:.6f}, "
              f"east={grid_bbox_latlon[2]:.6f}, west={grid_bbox_latlon[3]:.6f}")
    
    # Визначаємо source_crs якщо не задано
    if source_crs is None:
        try:
            # Конвертуємо bbox в UTM для визначення CRS
            bbox_utm_result = bbox_latlon_to_utm(*grid_bbox_latlon)
            source_crs = bbox_utm_result[4]  # CRS з результату
        except Exception as e:
            print(f"[WARN] Не вдалося визначити source_crs: {e}, використовується локальна нормалізація")
            return None, 0.0
    
    # Створюємо регулярну сітку точок для семплінгу висот по всій сітці
    # Використовуємо більш розріджену сітку для швидкості
    north, south, east, west = grid_bbox_latlon
    
    # Розраховуємо кількість точок на основі кількості зон
    # Для великої кількості зон використовуємо менше точок на зону
    total_zones = len(zones)
    if total_zones > 20:
        sample_points_per_zone = max(9, sample_points_per_zone // 2)  # Зменшуємо для великих сіток
    
    # Створюємо сітку точок для семплінгу.
    # CRITICAL (stitching): якщо elevation_ref_m завищений (через рідкий семплінг),
    # окремі зони отримують негативні Z і експорт потім "нульовить" кожну плитку по-різному -> шви.
    # Тому для малих батчів робимо густіший семплінг, щоб стабільно знайти реальний мінімум.
    grid_size = int(np.ceil(np.sqrt(total_zones * sample_points_per_zone)))
    grid_size = max(20, min(grid_size, 80))  # Мінімум 20x20, максимум 80x80
    
    # Створюємо регулярну сітку координат (lat/lon)
    lons = np.linspace(west, east, grid_size)
    lats = np.linspace(south, north, grid_size)
    X_lon, Y_lat = np.meshgrid(lons, lats)
    
    # Конвертуємо в UTM для семплінгу
    try:
        # ВИПРАВЛЕННЯ: Використовуємо глобальний центр для конвертації (якщо доступний)
        if global_center is not None:
            # Використовуємо глобальний центр для конвертації
            xs_utm_flat = []
            ys_utm_flat = []
            for lon, lat in zip(X_lon.flatten(), Y_lat.flatten()):
                try:
                    x_utm, y_utm = global_center.to_utm(lon, lat)
                    xs_utm_flat.append(x_utm)
                    ys_utm_flat.append(y_utm)
                except Exception:
                    # Fallback: використовуємо центр bbox
                    center_lat = (north + south) / 2.0
                    center_lon = (east + west) / 2.0
                    x_utm, y_utm = global_center.to_utm(center_lon, center_lat)
                    xs_utm_flat.append(x_utm)
                    ys_utm_flat.append(y_utm)
            
            X_utm = np.array(xs_utm_flat).reshape(X_lon.shape)
            Y_utm = np.array(ys_utm_flat).reshape(Y_lat.shape)
        else:
            # Fallback: використовуємо source_crs для конвертації
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", source_crs, always_xy=True)
            
            # Конвертуємо координати сітки в UTM
            xs_utm_flat = []
            ys_utm_flat = []
            for lon, lat in zip(X_lon.flatten(), Y_lat.flatten()):
                try:
                    x_utm, y_utm = transformer.transform(lon, lat)
                    xs_utm_flat.append(x_utm)
                    ys_utm_flat.append(y_utm)
                except Exception:
                    # Fallback: використовуємо центр bbox
                    center_lat = (north + south) / 2.0
                    center_lon = (east + west) / 2.0
                    x_utm, y_utm = transformer.transform(center_lon, center_lat)
                    xs_utm_flat.append(x_utm)
                    ys_utm_flat.append(y_utm)
            
            X_utm = np.array(xs_utm_flat).reshape(X_lon.shape)
            Y_utm = np.array(ys_utm_flat).reshape(Y_lat.shape)
        
    except Exception as e:
        print(f"[WARN] Помилка конвертації координат для семплінгу: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0
    
    # 1) Robust min (Terrarium): scan tiles to find true minimum in bbox (prevents seams).
    try:
        tile_min = _terrarium_min_elevation_in_bbox(grid_bbox_latlon, terrarium_zoom=terrarium_zoom)
    except Exception:
        tile_min = None

    # 2) Fallback: sample absolute heights for the grid bbox
    try:
        Z_abs = get_elevation_abs_meters_from_api(
            bbox_latlon=grid_bbox_latlon,
            X_meters=X_utm,
            Y_meters=Y_utm,
            source_crs=source_crs,
            terrarium_zoom=terrarium_zoom,
        )
        
        if Z_abs is None or not np.any(np.isfinite(Z_abs)):
            print("[WARN] Не вдалося отримати дані висот, використовується локальна нормалізація")
            return None, 0.0
        
        # Robust reference for stitching:
        # Terrarium can contain outlier low pixels (even after per-tile quantile), and using absolute MIN
        # makes elevation_ref_m too low -> all Z become huge positive -> base thickness explodes.
        vals = np.asarray(Z_abs, dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        vals = vals[(vals > -5000.0) & (vals < 9000.0)]
        if vals.size == 0:
            return None, 0.0

        sample_min = float(np.min(vals))
        # Use low quantile as a stable "sea-level-like" reference for the city bbox
        # (prevents tower bases while keeping relative relief consistent across tiles).
        sample_q = float(np.quantile(vals, 0.01))  # 1% quantile

        tile_q = None
        if tile_min is not None and np.isfinite(tile_min):
            # Reject tile minima that are far below sampled distribution (likely outliers / corrupt pixels)
            # Allow some margin for real valleys (30m is plenty for city tiles)
            if float(tile_min) >= (sample_q - 30.0):
                tile_q = float(tile_min)

        elevation_ref_m = float(min(sample_q, tile_q)) if tile_q is not None else float(sample_q)
        elevation_max = float(np.nanmax(Z_abs))
        elevation_mean = float(np.nanmean(Z_abs))
        
        print(f"[INFO] Глобальний elevation_ref: min={elevation_ref_m:.2f}м, max={elevation_max:.2f}м, mean={elevation_mean:.2f}м")
        
        # baseline_offset_m: глобальний зсув (в метрах у світі) який гарантує,
        # що всі Z_rel >= 0 для всіх зон у батчі.
        # Це критично, бо експорт зсуває кожну плитку по minZ -> якщо minZ різний,
        # зони не стикуються по висоті.
        # baseline_offset_m:
        # Historically we used this to "lift" all terrain so Z>=0 everywhere.
        # In practice Terrarium can contain rare bogus low pixels that force a huge baseline (hundreds of meters),
        # creating "tower bases" and making zones look wrong.
        #
        # We now clamp Z>=0 in `get_elevation_data()` for global mode, so we KEEP baseline at 0
        # and rely on clamping instead of shifting the whole city upward.
        baseline_offset_m = 0.0
        
        return elevation_ref_m, baseline_offset_m
        
    except Exception as e:
        print(f"[WARN] Помилка обчислення глобального elevation_ref: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0


def calculate_optimal_base_thickness(
    elevation_ref_m: Optional[float],
    zones: List[Dict],
    model_size_mm: float = 100.0,
    min_base_thickness_mm: float = 1.0,
    max_base_thickness_mm: float = 3.0,
) -> float:
    """
    Обчислює оптимальну товщину підложки (base) для всіх зон.
    Мінімізує товщину, але забезпечує стабільність моделей та ідеальне стикування зон.
    
    Args:
        elevation_ref_m: Глобальна базова висота (якщо None - використовується мінімальна)
        zones: Список зон для аналізу
        model_size_mm: Розмір моделі в міліметрах
        min_base_thickness_mm: Мінімальна товщина підложки (мм)
        max_base_thickness_mm: Максимальна товщина підложки (мм)
    
    Returns:
        Оптимальна товщина підложки в міліметрах
    """
    # КРИТИЧНО: Для синхронізованих зон використовуємо мінімальну товщину
    # Це забезпечує, що підложка не зайва, але достатня для стабільності
    # Всі зони мають однакову базову висоту (elevation_ref_m), тому підложка може бути тоншою
    
    if elevation_ref_m is not None:
        # Якщо є глобальний elevation_ref - всі зони синхронізовані
        # Використовуємо мінімальну товщину для ідеального стикування
        # ЗМЕНШЕНО: використовуємо ще меншу товщину для кращого вигляду
        optimal_thickness = max(min_base_thickness_mm * 0.8, 0.8)  # Мінімум 0.8мм
        
        # Для великої кількості зон (>10) можна трохи збільшити для стабільності
        if len(zones) > 10:
            optimal_thickness = min(optimal_thickness * 1.1, max_base_thickness_mm)
    else:
        # Якщо немає глобального elevation_ref - використовуємо трохи більшу товщину
        # для компенсації можливих різниць між зонами
        optimal_thickness = min(min_base_thickness_mm * 1.2, max_base_thickness_mm)
    
    # Адаптуємо до розміру моделі
    # Для великих моделей (>200мм) можна трохи збільшити товщину для стабільності
    if model_size_mm > 200:
        optimal_thickness = min(optimal_thickness * 1.15, max_base_thickness_mm)
    elif model_size_mm < 50:
        # Для малих моделей (<50мм) можна зменшити товщину
        optimal_thickness = max(optimal_thickness * 0.9, min_base_thickness_mm)
    
    # Адаптуємо до кількості зон
    # Для багатьох зон (>20) використовуємо мінімальну товщину для кращого стикування
    if len(zones) > 20:
        optimal_thickness = min(optimal_thickness, min_base_thickness_mm * 1.05)
    
    print(f"[INFO] Оптимальна товщина підложки: {optimal_thickness:.2f}мм "
          f"(зони: {len(zones)}, модель: {model_size_mm:.0f}мм, elevation_ref: {'є' if elevation_ref_m is not None else 'немає'})")
    return optimal_thickness

