"""
Сервіс для отримання даних висот (рельєфу) через API
Підтримує кілька джерел: OpenTopography (безкоштовний), Mapbox (потрібен ключ)
"""
import numpy as np
import requests
from typing import Tuple, Optional, Dict, Any, List
import warnings
import os

def _debug(msg: str):
    if (os.getenv("ELEVATION_DEBUG") or "").lower() in ["1", "true", "yes", "on"]:
        print(msg)


def get_elevation_from_api(
    bbox: Tuple[float, float, float, float],
    resolution: int = 100,
    api_type: str = "opentopography"
) -> Optional[np.ndarray]:
    """
    Отримує дані висот через API
    
    Args:
        bbox: (north, south, east, west) в градусах
        resolution: Роздільна здатність сітки
        api_type: Тип API ("opentopography", "mapbox", "google")
    
    Returns:
        Масив висот Z або None
    """
    north, south, east, west = bbox
    
    if api_type == "opentopography":
        return get_elevation_opentopography(bbox, resolution)
    elif api_type == "mapbox":
        return get_elevation_mapbox(bbox, resolution)
    else:
        return None


def get_elevation_opentopography(
    bbox: Tuple[float, float, float, float],
    resolution: int = 100
) -> Optional[np.ndarray]:
    """
    Отримує дані висот з OpenTopography API (безкоштовний, без API ключа)
    Використовує SRTM дані
    
    Args:
        bbox: (north, south, east, west) в градусах
        resolution: Роздільна здатність
    
    Returns:
        Масив висот
    """
    north, south, east, west = bbox
    
    try:
        # OpenTopography API endpoint для SRTM
        url = "https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTM_GL1/srtmgl1"
        
        # Створюємо сітку координат
        x = np.linspace(west, east, resolution)
        y = np.linspace(south, north, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Для OpenTopography потрібно робити запити по точках
        # Але це повільно, тому використаємо спрощений підхід
        # Або можна використати локальний DEM файл
        
        # Поки що повертаємо None - буде використано fallback
        return None
        
    except Exception as e:
        print(f"Помилка отримання даних з OpenTopography: {e}")
        return None


def get_elevation_mapbox(
    bbox: Tuple[float, float, float, float],
    resolution: int = 100,
    api_key: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Отримує дані висот з Mapbox Terrain API (потрібен API ключ)
    
    Args:
        bbox: (north, south, east, west) в градусах
        resolution: Роздільна здатність
        api_key: Mapbox API ключ (з змінної середовища MAPBOX_API_KEY)
    
    Returns:
        Масив висот
    """
    import os
    
    api_key = api_key or os.getenv("MAPBOX_API_KEY")
    if not api_key:
        print("Mapbox API ключ не знайдено. Використовується плоский рельєф.")
        return None
    
    north, south, east, west = bbox
    
    try:
        # Mapbox Terrain API
        # Формат: bbox=west,south,east,north
        url = f"https://api.mapbox.com/v4/mapbox.terrain-rgb/tilejson.json?access_token={api_key}"
        
        # Це складніше - потрібно завантажувати тайли та об'єднувати
        # Поки що повертаємо None
        return None
        
    except Exception as e:
        print(f"Помилка отримання даних з Mapbox: {e}")
        return None


def get_elevation_abs_meters_from_api(
    bbox_latlon: Tuple[float, float, float, float],
    X_meters: np.ndarray,
    Y_meters: np.ndarray,
    source_crs: Optional[object] = None,
    terrarium_zoom: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Отримує абсолютні висоти в метрах над рівнем моря з API (без нормалізації).
    Використовується для отримання реальних висот перед нормалізацією в terrain_generator.
    """
    # Default to terrarium tiles to avoid OpenTopoData rate limits.
    provider = (os.getenv("ELEVATION_PROVIDER") or "terrarium").lower()
    if provider in ["none", "synthetic", "fake"]:
        return None

    if source_crs is None:
        return None

    try:
        # 0) Локальний DEM (GeoTIFF) — найточніший варіант
        dem_path = os.getenv("DEM_PATH")
        if dem_path:
            Z = _get_elevation_from_geotiff(dem_path, X_meters, Y_meters, source_crs=source_crs)
            if Z is not None:
                _debug(f"[elevation] GeoTIFF '{dem_path}' ok: abs_range={float(np.nanmin(Z)):.2f}..{float(np.nanmax(Z)):.2f}m")
                return Z

        # Конвертація projected (UTM/метри) -> WGS84 для API
        from pyproj import Transformer

        transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)

        xs = X_meters.flatten().astype(float)
        ys = Y_meters.flatten().astype(float)
        lons, lats = transformer.transform(xs, ys)
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)

        if provider == "terrarium":
            from services.terrarium_tiles import TerrariumTileProvider

            zoom = int(terrarium_zoom if terrarium_zoom is not None else os.getenv("TERRARIUM_ZOOM", "14"))
            base_url = os.getenv("TERRARIUM_URL", "https://s3.amazonaws.com/elevation-tiles-prod/terrarium")
            cache_dir = os.getenv("TERRARIUM_CACHE_DIR", "cache/terrarium")
            timeout = float(os.getenv("TERRARIUM_TIMEOUT", "30"))

            _debug(f"[elevation] Terrarium tiles z={zoom}, points={lats.size}")
            provider_obj = TerrariumTileProvider(base_url=base_url, cache_dir=cache_dir, timeout=timeout)
            Z_flat = provider_obj.sample_points(lats, lons, z=zoom)
            if Z_flat is None or not np.any(np.isfinite(Z_flat)):
                _debug("[elevation] Terrarium failed -> fallback to synthetic")
                return None

        else:
            # OpenTopoData
            dataset = os.getenv("OPENTOPODATA_DATASET")
            if not dataset:
                north, south, east, west = bbox_latlon
                mid_lat = (north + south) / 2.0
                mid_lon = (east + west) / 2.0
                if 35.0 <= mid_lat <= 72.0 and -25.0 <= mid_lon <= 45.0:
                    dataset = "eudem25m"
                else:
                    dataset = "srtm30m"
            _debug(f"[elevation] OpenTopoData dataset={dataset}, points={lats.size}")

            Z_flat = _get_elevation_opentopodata_batch(lats, lons, dataset=dataset)
            if Z_flat is None:
                _debug("[elevation] OpenTopoData failed -> fallback to synthetic")
                return None

        Z = Z_flat.reshape(X_meters.shape)
        
        # Повертаємо абсолютні значення БЕЗ нормалізації
        zmin = float(np.nanmin(Z)) if np.any(~np.isnan(Z)) else 0.0
        zmax = float(np.nanmax(Z)) if np.any(~np.isnan(Z)) else 0.0
        _debug(f"[elevation] {provider} abs_range={zmin:.2f}..{zmax:.2f}m (absolute, not normalized)")

        # Заповнюємо NaN, але зберігаємо абсолютні значення
        Z = np.where(np.isnan(Z), zmin, Z)
        return Z
    except Exception as e:
        print(f"[WARN] elevation API failed: {e}")
        return None


def get_elevation_data_from_api(
    bbox_latlon: Tuple[float, float, float, float],
    X_meters: np.ndarray,
    Y_meters: np.ndarray,
    z_scale: float = 1.5,
    source_crs: Optional[object] = None,
    terrarium_zoom: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Отримує дані висот з API для заданої сітки координат
    
    Args:
        bbox_latlon: Bounding box (north, south, east, west) в градусах
        X_meters: 2D масив X координат в метрах
        Y_meters: 2D масив Y координат в метрах
        z_scale: Множник висоти
        
    Returns:
        Масив висот Z (2D, такий самий розмір як X_meters та Y_meters) або None
    """
    # Default to terrarium tiles to avoid OpenTopoData rate limits.
    provider = (os.getenv("ELEVATION_PROVIDER") or "terrarium").lower()
    if provider in ["none", "synthetic", "fake"]:
        return None

    if source_crs is None:
        return None

    try:
        # 0) Локальний DEM (GeoTIFF) — найточніший варіант, без лімітів API
        dem_path = os.getenv("DEM_PATH")
        if dem_path:
            Z = _get_elevation_from_geotiff(dem_path, X_meters, Y_meters, source_crs=source_crs)
            if Z is not None:
                zmin = float(np.nanmin(Z)) if np.any(~np.isnan(Z)) else 0.0
                zmax = float(np.nanmax(Z)) if np.any(~np.isnan(Z)) else 0.0
                _debug(f"[elevation] GeoTIFF '{dem_path}' ok: abs_range={zmin:.2f}..{zmax:.2f}m")
                Z = (Z - zmin) * float(z_scale)
                Z = np.where(np.isnan(Z), 0.0, Z)
                _debug(f"[elevation] GeoTIFF normalized: rel_range={float(np.min(Z)):.2f}..{float(np.max(Z)):.2f}m (z_scale={z_scale})")
                return Z

        # Конвертація projected (UTM/метри) -> WGS84 для API
        from pyproj import Transformer

        transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)

        xs = X_meters.flatten().astype(float)
        ys = Y_meters.flatten().astype(float)
        lons, lats = transformer.transform(xs, ys)
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)

        if provider == "terrarium":
            from services.terrarium_tiles import TerrariumTileProvider

            zoom = int(terrarium_zoom if terrarium_zoom is not None else os.getenv("TERRARIUM_ZOOM", "14"))
            base_url = os.getenv("TERRARIUM_URL", "https://s3.amazonaws.com/elevation-tiles-prod/terrarium")
            cache_dir = os.getenv("TERRARIUM_CACHE_DIR", "cache/terrarium")
            timeout = float(os.getenv("TERRARIUM_TIMEOUT", "30"))

            _debug(f"[elevation] Terrarium tiles z={zoom}, points={lats.size}")
            provider_obj = TerrariumTileProvider(base_url=base_url, cache_dir=cache_dir, timeout=timeout)
            Z_flat = provider_obj.sample_points(lats, lons, z=zoom)
            if Z_flat is None or not np.any(np.isfinite(Z_flat)):
                _debug("[elevation] Terrarium failed -> fallback to synthetic")
                return None

        else:
            # OpenTopoData: безкоштовний, без ключа (але може бути повільно/ліміти; робимо батчинг)
            dataset = os.getenv("OPENTOPODATA_DATASET")
            if not dataset:
                # Кращий дефолт для Європи
                north, south, east, west = bbox_latlon
                mid_lat = (north + south) / 2.0
                mid_lon = (east + west) / 2.0
                if 35.0 <= mid_lat <= 72.0 and -25.0 <= mid_lon <= 45.0:
                    dataset = "eudem25m"
                else:
                    dataset = "srtm30m"
            _debug(f"[elevation] OpenTopoData dataset={dataset}, points={lats.size}")

            Z_flat = _get_elevation_opentopodata_batch(lats, lons, dataset=dataset)
            if Z_flat is None:
                _debug("[elevation] OpenTopoData failed -> fallback to synthetic")
                return None

        Z = Z_flat.reshape(X_meters.shape)

        # Нормалізація: робимо висоту відносною до мінімуму в bbox (інакше буде "абсолютна" висота над морем)
        zmin = float(np.nanmin(Z)) if np.any(~np.isnan(Z)) else 0.0
        zmax = float(np.nanmax(Z)) if np.any(~np.isnan(Z)) else 0.0
        _debug(f"[elevation] {provider} abs_range={zmin:.2f}..{zmax:.2f}m")
        Z = (Z - zmin) * float(z_scale)
        _debug(f"[elevation] {provider} normalized rel_range={float(np.nanmin(Z)):.2f}..{float(np.nanmax(Z)):.2f}m (z_scale={z_scale})")

        # Заповнюємо NaN
        Z = np.where(np.isnan(Z), 0.0, Z)
        return Z
    except Exception as e:
        print(f"[WARN] elevation API failed: {e}")
        return None


_ELEV_CACHE: Dict[str, float] = {}


def _get_elevation_opentopodata_batch(lats: np.ndarray, lons: np.ndarray, dataset: str = "srtm30m") -> Optional[np.ndarray]:
    """
    Викликає OpenTopoData для набору точок.
    API лімітований, тому:
    - квантуємо координати (кеш)
    - робимо батчі по 100 точок
    """
    if lats.size == 0:
        return np.array([])

    base_url = os.getenv("OPENTOPODATA_URL", "https://api.opentopodata.org/v1")
    url = f"{base_url}/{dataset}"

    # Квантуємо для кешу (≈1–2м по широті/довготі)
    q_lat = np.round(lats, 5)
    q_lon = np.round(lons, 5)

    out = np.empty_like(q_lat, dtype=float)
    out.fill(np.nan)

    # Індекси які треба запитати
    to_fetch: List[int] = []
    for i in range(q_lat.size):
        key = f"{q_lat[i]:.5f},{q_lon[i]:.5f}"
        if key in _ELEV_CACHE:
            out[i] = _ELEV_CACHE[key]
        else:
            to_fetch.append(i)

    if not to_fetch:
        return out

    session = requests.Session()
    timeout = float(os.getenv("OPENTOPODATA_TIMEOUT", "30"))
    batch_size = int(os.getenv("OPENTOPODATA_BATCH", "100"))

    for start in range(0, len(to_fetch), batch_size):
        batch_idx = to_fetch[start:start + batch_size]
        locs = "|".join([f"{q_lat[i]:.5f},{q_lon[i]:.5f}" for i in batch_idx])
        try:
            resp = session.get(url, params={"locations": locs}, timeout=timeout)
            if resp.status_code != 200:
                _debug(f"[elevation] OpenTopoData HTTP {resp.status_code}: {resp.text[:200]}")
                return None
            data: Dict[str, Any] = resp.json()
            results = data.get("results") or []
            if len(results) != len(batch_idx):
                # Нестиковка відповіді
                _debug(f"[elevation] OpenTopoData mismatch results={len(results)} expected={len(batch_idx)}")
                return None
            for j, rec in enumerate(results):
                elev = rec.get("elevation", None)
                if elev is None:
                    continue
                val = float(elev)
                idx = batch_idx[j]
                out[idx] = val
                _ELEV_CACHE[f"{q_lat[idx]:.5f},{q_lon[idx]:.5f}"] = val
        except Exception:
            return None

    return out


def _get_elevation_from_geotiff(
    dem_path: str,
    X_meters: np.ndarray,
    Y_meters: np.ndarray,
    source_crs: object,
) -> Optional[np.ndarray]:
    """
    Семплить DEM GeoTIFF (Copernicus/SRTM/LiDAR) у будь-якій CRS:
    - переводимо точки з source_crs (UTM/метри) у CRS растру
    - rasterio.sample() повертає висоти
    """
    try:
        import rasterio
        from pyproj import Transformer

        if not os.path.exists(dem_path):
            _debug(f"[elevation] DEM_PATH not found: {dem_path}")
            return None

        with rasterio.open(dem_path) as src:
            raster_crs = src.crs
            if raster_crs is None:
                return None

            transformer = Transformer.from_crs(source_crs, raster_crs, always_xy=True)

            xs = X_meters.flatten().astype(float)
            ys = Y_meters.flatten().astype(float)
            rx, ry = transformer.transform(xs, ys)

            samples = list(src.sample(zip(rx, ry)))
            if not samples:
                return None

            arr = np.array(samples, dtype=float).reshape(-1)
            nodata = src.nodata
            if nodata is not None:
                arr = np.where(arr == float(nodata), np.nan, arr)

            Z = arr.reshape(X_meters.shape)
            return Z
    except Exception as e:
        _debug(f"[elevation] GeoTIFF sample failed: {e}")
        return None


def get_elevation_simple_terrain(
    X: np.ndarray,
    Y: np.ndarray,
    bbox: Tuple[float, float, float, float],
    z_scale: float = 1.5
) -> np.ndarray:
    """
    Генерує простий синтетичний рельєф для тестування
    (якщо API недоступний)
    
    Args:
        X: Масив X координат (в метрах, відносно центру)
        Y: Масив Y координат (в метрах, відносно центру)
        bbox: Bounding box (в градусах, для інформації)
        z_scale: Множник висоти
    
    Returns:
        Масив висот (в метрах)
    """
    # Простий хвильовий рельєф для демонстрації
    # У реальному застосуванні замініть на дані з API
    
    # Робимо координати відносними до центру області, щоб працювало і для UTM "великих" значень
    center_x = float(np.mean(X)) if X.size > 0 else 0.0
    center_y = float(np.mean(Y)) if Y.size > 0 else 0.0
    
    # Відстань від центру
    dx = X - center_x
    dy = Y - center_y
    distance = np.sqrt(dx**2 + dy**2)
    
    # Адаптивний масштаб на основі розміру області
    max_distance = np.max(distance) if distance.size > 0 else 1000
    scale = max(max_distance / 10, 100)  # Адаптивний масштаб
    
    # Хвильовий рельєф (висота в метрах) - збільшено амплітуду для кращої видимості
    Z = np.sin(distance / scale) * np.cos(dx / (scale * 0.6)) * 10 * z_scale  # Збільшено з 5 до 10
    
    # Додаємо базову висоту, щоб рельєф був видимий
    Z = Z + 2.0  # Збільшено з 1.0 до 2.0 метра базова висота
    
    # Додаємо додаткові деталі для кращої видимості
    Z = Z + np.sin(dx / (scale * 0.3)) * np.cos(dy / (scale * 0.3)) * 3 * z_scale
    
    return Z

