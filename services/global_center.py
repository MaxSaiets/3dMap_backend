"""
Сервіс для визначення та роботи з глобальним центром карти
Забезпечує єдину точку відліку (0,0) для всіх квадратів карти
"""
from typing import Tuple, Optional
from pyproj import CRS, Transformer
import math


class GlobalCenter:
    """
    Клас для роботи з глобальним центром карти
    Всі квадрати карти використовують цей центр як (0,0) для синхронізації
    """
    
    def __init__(
        self,
        center_lat: float,
        center_lon: float,
        utm_zone: Optional[int] = None,
    ):
        """
        Ініціалізує глобальний центр
        
        Args:
            center_lat: Широта глобального центру (WGS84)
            center_lon: Довгота глобального центру (WGS84)
            utm_zone: UTM зона (якщо None - визначається автоматично)
        """
        self.center_lat = float(center_lat)
        self.center_lon = float(center_lon)
        
        # Визначаємо UTM зону
        if utm_zone is None:
            self.utm_zone = math.floor((center_lon + 180) / 6) + 1
            self.utm_zone = max(1, min(60, self.utm_zone))
        else:
            self.utm_zone = utm_zone
        
        # Визначаємо UTM CRS
        is_south = center_lat < 0
        base_code = 32700 if is_south else 32600
        epsg_code = base_code + self.utm_zone
        self.utm_crs = CRS.from_epsg(epsg_code)
        self.wgs84_crs = CRS.from_epsg(4326)
        
        # Створюємо трансформери
        transformer_to_utm = Transformer.from_crs(self.wgs84_crs, self.utm_crs, always_xy=True)
        transformer_to_wgs84 = Transformer.from_crs(self.utm_crs, self.wgs84_crs, always_xy=True)
        
        # Конвертуємо центр в UTM метри
        self.center_x_utm, self.center_y_utm = transformer_to_utm.transform(center_lon, center_lat)
        
        # Зберігаємо трансформери
        self._to_utm = transformer_to_utm.transform
        self._to_wgs84 = transformer_to_wgs84.transform
        
        print(f"[GlobalCenter] Ініціалізовано: lat={center_lat:.6f}, lon={center_lon:.6f}, UTM zone={self.utm_zone}, center_utm=({self.center_x_utm:.2f}, {self.center_y_utm:.2f})")
    
    def to_utm(self, lon: float, lat: float) -> Tuple[float, float]:
        """
        Перетворює координати WGS84 в UTM метри
        
        Args:
            lon: Довгота (WGS84)
            lat: Широта (WGS84)
            
        Returns:
            Tuple (x, y) в UTM метрах
        """
        return self._to_utm(lon, lat)
    
    def to_wgs84(self, x: float, y: float) -> Tuple[float, float]:
        """
        Перетворює координати UTM метри в WGS84
        
        Args:
            x: X координата в UTM метрах
            y: Y координата в UTM метрах
            
        Returns:
            Tuple (lon, lat) в WGS84
        """
        return self._to_wgs84(x, y)
    
    def to_local(self, x_utm: float, y_utm: float) -> Tuple[float, float]:
        """
        Перетворює UTM координати в локальні (відносно глобального центру)
        
        Args:
            x_utm: X координата в UTM метрах
            y_utm: Y координата в UTM метрах
            
        Returns:
            Tuple (x_local, y_local) - координати відносно глобального центру (0,0)
        """
        x_local = x_utm - self.center_x_utm
        y_local = y_utm - self.center_y_utm
        return (x_local, y_local)
    
    def from_local(self, x_local: float, y_local: float) -> Tuple[float, float]:
        """
        Перетворює локальні координати (відносно глобального центру) в UTM
        
        Args:
            x_local: X координата відносно глобального центру
            y_local: Y координата відносно глобального центру
            
        Returns:
            Tuple (x_utm, y_utm) в UTM метрах
        """
        x_utm = x_local + self.center_x_utm
        y_utm = y_local + self.center_y_utm
        return (x_utm, y_utm)
    
    def bbox_to_local(
        self,
        north: float,
        south: float,
        east: float,
        west: float
    ) -> Tuple[float, float, float, float]:
        """
        Конвертує bbox з WGS84 в локальні координати (відносно глобального центру)
        
        Args:
            north: Північна межа (широта)
            south: Південна межа (широта)
            east: Східна межа (довгота)
            west: Західна межа (довгота)
            
        Returns:
            Tuple (minx_local, miny_local, maxx_local, maxy_local) в локальних координатах
        """
        # Конвертуємо кути bbox в UTM
        corners_lon = [west, east, west, east]
        corners_lat = [north, north, south, south]
        corners_x, corners_y = self.to_utm(corners_lon, corners_lat)
        
        # Знаходимо межі в UTM
        minx_utm = float(min(corners_x))
        maxx_utm = float(max(corners_x))
        miny_utm = float(min(corners_y))
        maxy_utm = float(max(corners_y))
        
        # Конвертуємо в локальні координати
        minx_local, miny_local = self.to_local(minx_utm, miny_utm)
        maxx_local, maxy_local = self.to_local(maxx_utm, maxy_utm)
        
        return (minx_local, miny_local, maxx_local, maxy_local)
    
    def get_center_utm(self) -> Tuple[float, float]:
        """
        Повертає координати глобального центру в UTM метрах
        
        Returns:
            Tuple (center_x_utm, center_y_utm)
        """
        return (self.center_x_utm, self.center_y_utm)
    
    def get_center_wgs84(self) -> Tuple[float, float]:
        """
        Повертає координати глобального центру в WGS84
        
        Returns:
            Tuple (center_lat, center_lon)
        """
        return (self.center_lat, self.center_lon)
    
    def get_utm_crs(self) -> CRS:
        """
        Повертає UTM CRS для цього центру
        
        Returns:
            CRS об'єкт
        """
        return self.utm_crs


# Глобальна змінна для зберігання центру (можна перевизначити через API)
_global_center: Optional[GlobalCenter] = None
# Глобальний bbox (WGS84) для DEM-семплінгу у batch режимі (north, south, east, west).
# Важливо: всі зони повинні семплити висоти з однакового bbox, інакше на межах можуть бути NaN/різні тайли.
_global_dem_bbox_latlon: Optional[Tuple[float, float, float, float]] = None


def set_global_center(
    center_lat: float,
    center_lon: float,
    utm_zone: Optional[int] = None,
) -> GlobalCenter:
    """
    Встановлює глобальний центр карти
    
    Args:
        center_lat: Широта глобального центру (WGS84)
        center_lon: Довгота глобального центру (WGS84)
        utm_zone: UTM зона (якщо None - визначається автоматично)
        
    Returns:
        GlobalCenter об'єкт
    """
    global _global_center
    _global_center = GlobalCenter(center_lat, center_lon, utm_zone)
    return _global_center


def get_global_center() -> Optional[GlobalCenter]:
    """
    Отримує поточний глобальний центр
    
    Returns:
        GlobalCenter об'єкт або None, якщо не встановлено
    """
    return _global_center


def set_global_dem_bbox_latlon(bbox_latlon: Tuple[float, float, float, float]) -> None:
    """
    Зберігає глобальний bbox (WGS84) для DEM-семплінгу у batch режимі.
    Формат: (north, south, east, west)
    """
    global _global_dem_bbox_latlon
    _global_dem_bbox_latlon = tuple(map(float, bbox_latlon))


def get_global_dem_bbox_latlon() -> Optional[Tuple[float, float, float, float]]:
    """
    Повертає глобальний bbox (WGS84) для DEM-семплінгу у batch режимі, якщо встановлено.
    """
    return _global_dem_bbox_latlon


def get_or_create_global_center(
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    bbox_latlon: Optional[Tuple[float, float, float, float]] = None,
) -> GlobalCenter:
    """
    Отримує існуючий глобальний центр або створює новий
    
    Args:
        center_lat: Широта центру (якщо None - обчислюється з bbox)
        center_lon: Довгота центру (якщо None - обчислюється з bbox)
        bbox_latlon: Bounding box (north, south, east, west) для обчислення центру
        
    Returns:
        GlobalCenter об'єкт
    """
    global _global_center
    
    if _global_center is not None:
        return _global_center
    
    # Якщо центр не встановлено, обчислюємо його
    if center_lat is None or center_lon is None:
        if bbox_latlon is not None:
            north, south, east, west = bbox_latlon
            center_lat = (north + south) / 2.0
            center_lon = (east + west) / 2.0
        else:
            raise ValueError("Потрібно вказати center_lat/center_lon або bbox_latlon")
    
    _global_center = GlobalCenter(center_lat, center_lon)
    return _global_center

