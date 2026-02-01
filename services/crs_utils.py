"""
Утиліти для роботи з системами координат (CRS)
Забезпечує правильне перетворення між WGS84 (Lat/Lon) та UTM проекціями
"""
import math
from typing import Tuple, Callable, Optional
from pyproj import CRS, Transformer
from shapely.geometry import Point
from shapely.ops import transform


def get_utm_crs_from_latlon(lat: float, lon: float) -> CRS:
    """
    Визначає правильну UTM зону для заданої точки.
    Це дозволяє працювати в метрах з максимальною точністю.
    
    Args:
        lat: Широта в градусах (WGS84)
        lon: Довгота в градусах (WGS84)
        
    Returns:
        CRS об'єкт для UTM зони
    """
    # UTM зони: кожні 6 градусів довготи
    zone_number = math.floor((lon + 180) / 6) + 1
    # Обмежуємо зону в межах 1-60
    zone_number = max(1, min(60, zone_number))
    
    # Визначаємо північ/південь
    is_south = lat < 0
    
    # EPSG коди для UTM: 326xx (Північ) або 327xx (Південь)
    base_code = 32700 if is_south else 32600
    epsg_code = base_code + zone_number
    
    return CRS.from_epsg(epsg_code)


def get_transformers(lat: float, lon: float) -> Tuple[Callable, Callable, CRS]:
    """
    Створює трансформери для перетворення між WGS84 та UTM.
    
    Args:
        lat: Центральна широта області (WGS84)
        lon: Центральна довгота області (WGS84)
        
    Returns:
        Tuple (to_utm, to_wgs84, utm_crs):
        - to_utm: функція для перетворення (lon, lat) -> (x, y) в метрах
        - to_wgs84: функція для перетворення (x, y) -> (lon, lat)
        - utm_crs: CRS об'єкт UTM зони
    """
    utm_crs = get_utm_crs_from_latlon(lat, lon)
    wgs84_crs = CRS.from_epsg(4326)
    
    # always_xy=True означає порядок (lon, lat) або (x, y)
    transformer_to_utm = Transformer.from_crs(wgs84_crs, utm_crs, always_xy=True)
    transformer_to_wgs84 = Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)
    
    def to_utm(lon_or_x, lat_or_y):
        """Перетворює (lon, lat) -> (x, y) в метрах"""
        return transformer_to_utm.transform(lon_or_x, lat_or_y)
    
    def to_wgs84(x, y):
        """Перетворює (x, y) в метрах -> (lon, lat)"""
        return transformer_to_wgs84.transform(x, y)
    
    return to_utm, to_wgs84, utm_crs


def bbox_latlon_to_utm(
    north: float, 
    south: float, 
    east: float, 
    west: float
) -> Tuple[float, float, float, float, CRS, Callable, Callable]:
    """
    Конвертує bounding box з WGS84 (Lat/Lon) в UTM метри.
    
    Args:
        north: Північна межа (широта)
        south: Південна межа (широта)
        east: Східна межа (довгота)
        west: Західна межа (довгота)
        
    Returns:
        Tuple (minx, miny, maxx, maxy, utm_crs, to_utm, to_wgs84):
        - minx, miny, maxx, maxy: BBox в метрах (UTM)
        - utm_crs: CRS об'єкт UTM зони
        - to_utm: функція для перетворення (lon, lat) -> (x, y)
        - to_wgs84: функція для перетворення (x, y) -> (lon, lat)
    """
    # Визначаємо центр області для вибору UTM зони
    center_lat = (north + south) / 2.0
    center_lon = (east + west) / 2.0
    
    # Отримуємо трансформери
    to_utm, to_wgs84, utm_crs = get_transformers(center_lat, center_lon)
    
    # Конвертуємо кути bbox в метри
    # Важливо: конвертуємо всі 4 кути, щоб отримати правильні межі
    corners_lon = [west, east, west, east]
    corners_lat = [north, north, south, south]
    
    corners_x, corners_y = to_utm(corners_lon, corners_lat)
    
    # Знаходимо мінімальні та максимальні значення
    minx = float(min(corners_x))
    maxx = float(max(corners_x))
    miny = float(min(corners_y))
    maxy = float(max(corners_y))
    
    return minx, miny, maxx, maxy, utm_crs, to_utm, to_wgs84


def transform_geometry_to_utm(geometry, to_utm: Callable):
    """
    Перетворює Shapely геометрію з WGS84 в UTM.
    
    Args:
        geometry: Shapely геометрія в WGS84
        to_utm: Функція перетворення (lon, lat) -> (x, y)
        
    Returns:
        Перетворена геометрія в UTM
    """
    if geometry is None or geometry.is_empty:
        return geometry
    
    # shapely.ops.transform очікує функцію, яка приймає (x, y) і повертає (x', y')
    # Наша to_utm приймає (lon, lat) і повертає (x, y)
    # Для Shapely геометрій в WGS84: x = lon, y = lat
    def transform_func(x, y, z=None):
        """Трансформер для Shapely"""
        x_new, y_new = to_utm(x, y)
        if z is not None:
            return (x_new, y_new, z)
        return (x_new, y_new)
    
    return transform(transform_func, geometry)

