import numpy as np
import time
from typing import Tuple, Optional, Iterable
from shapely.geometry.base import BaseGeometry

def _iter_polygons(geom: BaseGeometry):
    """Yield Polygon parts from Polygon/MultiPolygon/GeometryCollection-ish inputs."""
    if geom is None or geom.is_empty:
        return
    gt = getattr(geom, "geom_type", None)
    if gt == "Polygon":
        yield geom
    elif gt == "MultiPolygon":
        for p in geom.geoms:
            yield p
    elif gt == "GeometryCollection":
        for g in geom.geoms:
            yield from _iter_polygons(g)

def flatten_heightfield_under_buildings(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    building_geometries: Iterable[BaseGeometry],
    # quantile: float = 0.90, # Unused, removed per review
    all_touched: bool = True,
    min_cells: int = 2,
) -> np.ndarray:
    """
    Flattens terrain under buildings using median height for stability.
    Ensures buildings sit flat on the terrain.
    """
    Z_out = np.array(Z, dtype=float, copy=True)
    if X.ndim != 2 or Y.ndim != 2 or Z_out.ndim != 2: return Z_out
    rows, cols = Z_out.shape
    if rows < 2 or cols < 2: return Z_out

    # Assertion for Axis-Aligned Grid (Rasterio transform reliability)
    # Check first row X is monotonic and uniform
    # Check first col Y is monotonic and uniform
    try:
        if not (np.allclose(X[0], X[1]) if rows > 1 else True) and np.allclose(Y[:, 0], Y[:, 1]) if cols > 1 else True:
            # Not strictly axis aligned or meshgrid might be rotated?
            # Rasterio transform assumes axis aligned bbox.
            pass 
        # Ideally: assert np.allclose(X[0,1]-X[0,0], X[0,2]-X[0,1])
    except: pass

    minx, maxx, miny, maxy = np.min(X), np.max(X), np.min(Y), np.max(Y)

    try:
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds
        transform = from_bounds(minx, miny, maxx, maxy, cols, rows)
        
        for g in building_geometries:
            for poly in _iter_polygons(g):
                if not poly or poly.is_empty: continue
                try:
                    if not poly.is_valid: poly = poly.buffer(0)
                except: continue
                if poly.is_empty: continue

                mask = rasterize([(poly, 1)], out_shape=(rows, cols), transform=transform, 
                                 fill=0, dtype="uint8", all_touched=all_touched).astype(bool)
                
                # min_cells: avoids flattening single-pixel noise or slivers
                if mask.sum() < min_cells: continue
                h = Z_out[mask]
                h = h[np.isfinite(h)]
                if h.size < min_cells: continue
                
                ref = float(np.median(h))
                Z_out[mask] = ref
        return Z_out
    except Exception as e:
        print(f"[WARN] flatten_buildings failed: {e}")
        return Z_out

def flatten_heightfield_under_polygons(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    geometries: Iterable[BaseGeometry],
    quantile: float = 0.50,
    all_touched: bool = True,
    min_cells: int = 2,
) -> np.ndarray:
    """Generic flattening using quantile."""
    Z_out = np.array(Z, dtype=float, copy=True)
    if X.ndim != 2 or Y.ndim != 2: return Z_out
    rows, cols = Z_out.shape
    
    try:
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds
        transform = from_bounds(np.min(X), np.min(Y), np.max(X), np.max(Y), cols, rows)
        
        for g in geometries:
            for poly in _iter_polygons(g):
                if not poly or poly.is_empty: continue
                try:
                    if not poly.is_valid: poly = poly.buffer(0)
                except: continue
                
                mask = rasterize([(poly, 1)], out_shape=(rows, cols), transform=transform, 
                                 fill=0, dtype="uint8", all_touched=all_touched).astype(bool)
                if mask.sum() < min_cells: continue
                h = Z_out[mask]
                h = h[np.isfinite(h)]
                if h.size < min_cells: continue
                
                ref = float(np.quantile(h, float(quantile)))
                Z_out[mask] = ref
        return Z_out
    except: return Z_out

def depress_heightfield_under_polygons(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    geometries: Iterable[BaseGeometry],
    depth: float,
    min_floor: Optional[float] = None,
    quantile: float = 0.50,
    all_touched: bool = True,
    min_cells: int = 2,
) -> np.ndarray:
    """Carves a depression (for water)."""
    Z_out = np.array(Z, dtype=float, copy=True)
    if depth <= 0: return Z_out
    
    try:
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds
        rows, cols = Z_out.shape
        transform = from_bounds(np.min(X), np.min(Y), np.max(X), np.max(Y), cols, rows)
        
        for g in geometries:
            for poly in _iter_polygons(g):
                if not poly or poly.is_empty: continue
                try: 
                    if not poly.is_valid: poly = poly.buffer(0)
                except: continue
                
                mask = rasterize([(poly, 1)], out_shape=(rows, cols), transform=transform, 
                                 fill=0, dtype="uint8", all_touched=all_touched).astype(bool)
                if mask.sum() < min_cells: continue
                h = Z_out[mask]
                h = h[np.isfinite(h)]
                if h.size < min_cells: continue
                
                surface = float(np.min(h)) if quantile <= 0.0 else float(np.quantile(h, quantile))
                val = surface - depth
                Z_out[mask] = val
                if min_floor is not None:
                     Z_out[mask] = np.maximum(Z_out[mask], min_floor)
        return Z_out
    except: return Z_out

def get_elevation_data(
    X: np.ndarray,
    Y: np.ndarray,
    latlon_bbox: Optional[Tuple[float, float, float, float]],
    z_scale: float,
    source_crs: Optional[object] = None,
    terrarium_zoom: Optional[int] = None,
    elevation_ref_m: Optional[float] = None,
    baseline_offset_m: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """
    Оптимізоване отримання даних висот рельєфу з валідацією та очищенням.
    """
    total_start = time.time()
    print(f"[HEIGHTMAP] {'='*80}")
    print(f"[HEIGHTMAP] ПОЧАТОК get_elevation_data")
    print(f"[HEIGHTMAP] Параметри:")
    print(f"[HEIGHTMAP]   - Форма X/Y: {X.shape}")
    print(f"[HEIGHTMAP]   - z_scale: {z_scale}")
    print(f"[HEIGHTMAP]   - terrarium_zoom: {terrarium_zoom}")
    print(f"[HEIGHTMAP]   - elevation_ref_m: {elevation_ref_m}")
    print(f"[HEIGHTMAP]   - baseline_offset_m: {baseline_offset_m}")
    
    from services.elevation_api import get_elevation_abs_meters_from_api, get_elevation_simple_terrain
    
    # Валідація входу
    validation_start = time.time()
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    z_scale = float(z_scale)
    baseline_offset_m = float(baseline_offset_m)
    
    # Перевірка форми масивів
    if X.shape != Y.shape:
        raise ValueError(f"X та Y мають різні форми: {X.shape} vs {Y.shape}")
    print(f"[TIMING] Валідація входу: {time.time() - validation_start:.3f} сек")
    
    # Спробувати отримати CRS якщо не задано
    crs_start = time.time()
    Z_abs = None
    
    if latlon_bbox and not source_crs:
        try:
            from services.crs_utils import bbox_latlon_to_utm
            *_, utm_crs, _, _ = bbox_latlon_to_utm(*latlon_bbox)
            source_crs = utm_crs
            print(f"[HEIGHTMAP] Отримано CRS: {utm_crs}")
        except Exception as e:
            print(f"[WARN] Не вдалося отримати CRS: {e}")
    print(f"[TIMING] Отримання CRS: {time.time() - crs_start:.3f} сек")

    # Отримання абсолютних висот з API
    api_start = time.time()
    if latlon_bbox and source_crs:
        print(f"[HEIGHTMAP] Отримання висот з API...")
        try:
            Z_abs = get_elevation_abs_meters_from_api(latlon_bbox, X, Y, source_crs, terrarium_zoom)
            if Z_abs is not None:
                print(f"[HEIGHTMAP] Успішно отримано висоти з API")
            else:
                print(f"[WARN] API повернув None")
        except Exception as e:
            print(f"[WARN] Помилка отримання висот з API: {e}")
            Z_abs = None
    else:
        print(f"[HEIGHTMAP] Пропущено API (latlon_bbox={latlon_bbox is not None}, source_crs={source_crs is not None})")
    print(f"[TIMING] Отримання з API: {time.time() - api_start:.3f} сек")
    
    # Fallback до простого рельєфу
    if Z_abs is None:
        fallback_start = time.time()
        print(f"[HEIGHTMAP] Використання fallback (простий рельєф)...")
        try:
            Z_rel = get_elevation_simple_terrain(X, Y, (0,0,0,0), z_scale)
            Z_rel = np.asarray(Z_rel, dtype=np.float64)
            Z_rel = Z_rel + baseline_offset_m
            print(f"[TIMING] Fallback рельєф: {time.time() - fallback_start:.3f} сек")
            total_time = time.time() - total_start
            print(f"[TIMING] Загальний час get_elevation_data: {total_time:.3f} сек")
            print(f"[HEIGHTMAP] {'='*80}")
            return Z_rel, 0.0
        except Exception as e:
            print(f"[ERROR] Помилка fallback рельєфу: {e}")
            # Останній fallback - плоский рельєф
            return np.zeros_like(X, dtype=np.float64), 0.0

    # Обробка абсолютних висот
    process_start = time.time()
    Z_abs = np.asarray(Z_abs, dtype=np.float64)
    
    # Статистика абсолютних висот
    abs_stats_start = time.time()
    finite_abs = np.isfinite(Z_abs)
    if np.any(finite_abs):
        abs_min = float(np.nanmin(Z_abs[finite_abs]))
        abs_max = float(np.nanmax(Z_abs[finite_abs]))
        abs_mean = float(np.nanmean(Z_abs[finite_abs]))
        print(f"[HEIGHTMAP] Статистика абсолютних висот:")
        print(f"[HEIGHTMAP]   - Мін: {abs_min:.3f}m")
        print(f"[HEIGHTMAP]   - Макс: {abs_max:.3f}m")
        print(f"[HEIGHTMAP]   - Середнє: {abs_mean:.3f}m")
        print(f"[HEIGHTMAP]   - Валідних: {np.sum(finite_abs)}/{Z_abs.size}")
    
    # Перевірка на валідні дані
    if not np.any(finite_abs):
        print(f"[WARN] Всі висоти NaN/Inf, використовую fallback")
        try:
            Z_rel = get_elevation_simple_terrain(X, Y, (0,0,0,0), z_scale)
            Z_rel = np.asarray(Z_rel, dtype=np.float64) + baseline_offset_m
            return Z_rel, 0.0
        except:
            return np.zeros_like(X, dtype=np.float64), 0.0
    print(f"[TIMING] Статистика абсолютних висот: {time.time() - abs_stats_start:.3f} сек")
    
    # Конвертація до відносних висот
    convert_start = time.time()
    if elevation_ref_m is not None and np.isfinite(elevation_ref_m):
        # Використовуємо задану референсну висоту
        print(f"[HEIGHTMAP] Конвертація з elevation_ref_m: {elevation_ref_m:.3f}m")
        Z_rel = (Z_abs - float(elevation_ref_m)) * z_scale
    else:
        # Використовуємо мінімальну висоту як референс
        zmin = np.nanmin(Z_abs) if np.any(finite_abs) else 0.0
        print(f"[HEIGHTMAP] Конвертація з мінімальною висотою: {zmin:.3f}m")
        Z_rel = (Z_abs - zmin) * z_scale
    print(f"[TIMING] Конвертація до відносних висот: {time.time() - convert_start:.3f} сек")

    # Додавання baseline offset
    if baseline_offset_m != 0.0:
        Z_rel = Z_rel + baseline_offset_m
        print(f"[HEIGHTMAP] Додано baseline_offset: {baseline_offset_m:.3f}m")

    # Обмеження негативних значень якщо використовуємо elevation_ref_m
    if elevation_ref_m is not None:
        negative_count = np.sum(Z_rel < 0)
        if negative_count > 0:
            print(f"[HEIGHTMAP] Обмежено {negative_count} негативних значень до 0")
        Z_rel = np.maximum(Z_rel, 0.0)

    # Заповнення NaN значень
    fill_start = time.time()
    finite_mask = np.isfinite(Z_rel)
    if not np.all(finite_mask):
        nan_count = np.sum(~finite_mask)
        if np.any(finite_mask):
            # Використовуємо мінімальне значення для заповнення
            fill_value = np.nanmin(Z_rel[finite_mask])
        else:
            fill_value = 0.0
        Z_rel = np.where(finite_mask, Z_rel, fill_value)
        print(f"[WARN] Заповнено {nan_count} NaN значень висот значенням {fill_value:.2f}")
    print(f"[TIMING] Заповнення NaN: {time.time() - fill_start:.3f} сек")
    
    # Перевірка на екстремальні значення
    extreme_start = time.time()
    if np.any(np.abs(Z_rel) > 1e6):
        extreme_count = np.sum(np.abs(Z_rel) > 1e6)
        print(f"[WARN] Знайдено {extreme_count} екстремальних значень висот, обмежую...")
        z_median = np.nanmedian(Z_rel)
        z_std = np.nanstd(Z_rel)
        z_max = z_median + 10 * z_std
        z_min = z_median - 10 * z_std
        Z_rel = np.clip(Z_rel, z_min, z_max)
        print(f"[HEIGHTMAP] Обмежено до діапазону: [{z_min:.3f}, {z_max:.3f}]")
    print(f"[TIMING] Перевірка екстремальних значень: {time.time() - extreme_start:.3f} сек")
    
    # Фінальна статистика
    final_stats_start = time.time()
    if np.any(np.isfinite(Z_rel)):
        rel_min = float(np.nanmin(Z_rel))
        rel_max = float(np.nanmax(Z_rel))
        rel_mean = float(np.nanmean(Z_rel))
        rel_range = rel_max - rel_min
        print(f"[HEIGHTMAP] Фінальна статистика відносних висот:")
        print(f"[HEIGHTMAP]   - Мін: {rel_min:.3f}m")
        print(f"[HEIGHTMAP]   - Макс: {rel_max:.3f}m")
        print(f"[HEIGHTMAP]   - Середнє: {rel_mean:.3f}m")
        print(f"[HEIGHTMAP]   - Діапазон: {rel_range:.3f}m")
    print(f"[TIMING] Фінальна статистика: {time.time() - final_stats_start:.3f} сек")
    
    total_time = time.time() - total_start
    print(f"[TIMING] Загальний час get_elevation_data: {total_time:.3f} сек")
    print(f"[TIMING]   - API: {time.time() - api_start:.3f} сек")
    print(f"[TIMING]   - Обробка: {time.time() - process_start:.3f} сек")
    print(f"[HEIGHTMAP] {'='*80}")
    
    return Z_rel, float(zmin) if 'zmin' in locals() and zmin is not None else 0.0
