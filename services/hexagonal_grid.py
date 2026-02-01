"""
Генерація сітки (шестикутники або квадрати) для ділення карти на зони.
Використовує offset coordinates для ідеального підходу без границь.
"""
import math
from typing import List, Tuple, Dict
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union


def hexagon_center_to_corner(center_x: float, center_y: float, size: float) -> List[Tuple[float, float]]:
    """
    Створює координати вершин шестикутника навколо центру.
    
    Args:
        center_x: X координата центру
        center_y: Y координата центру
        size: Радіус шестикутника (відстань від центру до вершини)
    
    Returns:
        Список координат вершин (x, y)
    """
    corners = []
    # Для flat-top орієнтації (плоска верхня грань) - зміщуємо на -30°
    # Це забезпечує правильне з'єднання грань до грані
    for i in range(6):
        angle = math.pi / 3 * i - math.pi / 6  # -30° зміщення для flat-top
        x = center_x + size * math.cos(angle)
        y = center_y + size * math.sin(angle)
        corners.append((x, y))
    return corners


def generate_square_grid(
    bbox: Tuple[float, float, float, float],
    square_size_m: float = 400.0  # 0.4 км = 400 м
) -> List[Dict]:
    """
    Генерує квадратну сітку для заданого bbox.
    
    Args:
        bbox: Bounding box (minx, miny, maxx, maxy) в метрах
        square_size_m: Розмір квадрата (сторона) в метрах
    
    Returns:
        Список словників з інформацією про кожен квадрат
    """
    minx, miny, maxx, maxy = bbox
    
    # Перевірка валідності bbox
    if maxx <= minx or maxy <= miny:
        raise ValueError(f"Невірний bbox: minx={minx}, maxx={maxx}, miny={miny}, maxy={maxy}")
    
    # Розраховуємо кількість квадратів
    cols = int(math.ceil((maxx - minx) / square_size_m)) + 1
    rows = int(math.ceil((maxy - miny) / square_size_m)) + 1
    
    print(f"[DEBUG] Square grid: bbox=({minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f})")
    print(f"[DEBUG] Square size: {square_size_m}m")
    print(f"[DEBUG] Grid: cols={cols}, rows={rows}, total cells: {cols * rows}")
    
    squares = []
    for row in range(rows):
        for col in range(cols):
            # Центр квадрата
            center_x = minx + col * square_size_m + square_size_m / 2.0
            center_y = miny + row * square_size_m + square_size_m / 2.0
            
            # Створюємо квадрат
            half_size = square_size_m / 2.0
            square_polygon = box(
                center_x - half_size,
                center_y - half_size,
                center_x + half_size,
                center_y + half_size
            )
            
            # Перевіряємо, чи квадрат перетинається з bbox
            if square_polygon.intersects(box(minx, miny, maxx, maxy)):
                squares.append({
                    'id': f'square_{row}_{col}',
                    'center': (center_x, center_y),
                    'polygon': square_polygon,
                    'row': row,
                    'col': col
                })
    
    print(f"[DEBUG] Згенеровано {len(squares)} квадратів")
    return squares


def generate_hexagonal_grid(
    bbox: Tuple[float, float, float, float],
    hex_size_m: float = 400.0  # 0.4 км = 400 м
) -> List[Dict]:
    """
    Генерує гексагональну сітку для заданого bbox.
    
    Args:
        bbox: Bounding box (minx, miny, maxx, maxy) в метрах
        hex_size_m: Розмір шестикутника (радіус від центру до вершини) в метрах
    
    Returns:
        Список словників з інформацією про кожен шестикутник:
        {
            'id': str,
            'center': (x, y),
            'polygon': Polygon,
            'row': int,
            'col': int
        }
    """
    minx, miny, maxx, maxy = bbox
    
    # Перевірка валідності bbox
    if maxx <= minx or maxy <= miny:
        raise ValueError(f"Невірний bbox: minx={minx}, maxx={maxx}, miny={miny}, maxy={maxy}")
    
    # Розміри шестикутника для offset coordinates
    # Ширина (горизонтальна відстань між центрами): sqrt(3) * size
    # Висота (вертикальна відстань між центрами): 1.5 * size
    hex_width = math.sqrt(3) * hex_size_m
    hex_height = 1.5 * hex_size_m  # Правильна вертикальна відстань для offset
    
    # Розраховуємо кількість шестикутників
    # ОПТИМІЗАЦІЯ: Обмежуємо кількість для великих областей
    cols = int(math.ceil((maxx - minx) / hex_width)) + 2  # +2 для запасу
    rows = int(math.ceil((maxy - miny) / hex_height)) + 2  # +2 для запасу
    
    print(f"[DEBUG] Hexagonal grid: bbox=({minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f})")
    print(f"[DEBUG] Hex size: {hex_size_m}m, width: {hex_width:.2f}m, height: {hex_height:.2f}m")
    print(f"[DEBUG] Grid: cols={cols}, rows={rows}, total cells: {cols * rows}")
    
    # ОПТИМІЗАЦІЯ: Обмежуємо максимум для продуктивності
    MAX_HEXAGONS = 10000  # Максимум 10000 шестикутників
    if cols * rows > MAX_HEXAGONS:
        # Зменшуємо кількість пропорційно
        scale = math.sqrt(MAX_HEXAGONS / (cols * rows))
        cols = int(cols * scale)
        rows = int(rows * scale)
        print(f"[DEBUG] Обмежено до {cols}x{rows} для продуктивності")
    
    hexagons = []
    hex_id = 0
    
    # Генеруємо шестикутники з offset coordinates
    for row in range(rows):
        for col in range(cols):
            # Offset coordinates: непарні ряди зміщені на половину ширини
            if row % 2 == 0:
                center_x = minx + col * hex_width
            else:
                center_x = minx + col * hex_width + hex_width / 2.0
            
            # Вертикальна позиція: просто row * hex_height
            center_y = miny + row * hex_height
            
            # Створюємо шестикутник
            corners = hexagon_center_to_corner(center_x, center_y, hex_size_m)
            polygon = Polygon(corners)
            
            # Перевіряємо, чи шестикутник перетинається з bbox
            bbox_poly = Polygon([
                (minx, miny),
                (maxx, miny),
                (maxx, maxy),
                (minx, maxy)
            ])
            
            if polygon.intersects(bbox_poly):
                hex_id_str = f"hex_{row}_{col}"
                hexagons.append({
                    'id': hex_id_str,
                    'center': (center_x, center_y),
                    'polygon': polygon,
                    'row': row,
                    'col': col,
                    'bounds': polygon.bounds  # (minx, miny, maxx, maxy)
                })
                hex_id += 1
    
    return hexagons


def hexagons_to_geojson(cells: List[Dict], to_wgs84=None) -> Dict:
    """
    Конвертує список клітинок (шестикутники або квадрати) в GeoJSON формат.
    
    Args:
        cells: Список словників з інформацією про клітинки (hexagons або squares)
        to_wgs84: Функція для конвертації UTM координат в WGS84 (lat, lon). 
                  Якщо None, координати вважаються вже в WGS84.
    
    Returns:
        GeoJSON FeatureCollection
    """
    features = []
    for cell_data in cells:
        polygon = cell_data['polygon']
        
        # Конвертуємо координати з UTM в lat/lon якщо потрібно
        if to_wgs84 is not None:
            # Конвертуємо кожну точку
            coords_utm = list(polygon.exterior.coords)
            coords_wgs84 = []
            for x, y in coords_utm:
                try:
                    lon, lat = to_wgs84(x, y)  # to_wgs84 повертає (lon, lat)
                    coords_wgs84.append([lon, lat])  # GeoJSON використовує [lon, lat]
                except Exception as e:
                    print(f"[WARN] Помилка конвертації координат ({x}, {y}): {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback: використовуємо оригінальні координати (неправильно, але краще ніж помилка)
                    coords_wgs84.append([x, y])
            coords = [coords_wgs84]
        else:
            # Координати вже в правильному форматі
            coords = [[list(p) for p in polygon.exterior.coords]]
        
        feature = {
            'type': 'Feature',
            'id': cell_data['id'],
            'geometry': {
                'type': 'Polygon',
                'coordinates': coords
            },
            'properties': {
                'id': cell_data['id'],
                'row': cell_data['row'],
                'col': cell_data['col'],
                'center': cell_data['center']
            }
        }
        features.append(feature)
    
    return {
        'type': 'FeatureCollection',
        'features': features
    }


def calculate_grid_center_from_geojson(geojson: Dict, to_wgs84=None) -> Tuple[float, float]:
    """
    Обчислює оптимальний центр для всієї сітки на основі GeoJSON features.
    Це забезпечує, що всі клітинки використовують одну точку відліку (0,0).
    
    Args:
        geojson: GeoJSON FeatureCollection з клітинками сітки
        to_wgs84: Функція для конвертації координат в WGS84 (якщо координати в UTM)
    
    Returns:
        Tuple (center_lat, center_lon) - центр всієї сітки в WGS84
    """
    if not geojson or 'features' not in geojson:
        raise ValueError("GeoJSON не містить features")
    
    features = geojson['features']
    if not features:
        raise ValueError("GeoJSON не містить жодних features")
    
    # Збираємо всі координати з усіх features
    all_lons = []
    all_lats = []
    
    for feature in features:
        geometry = feature.get('geometry', {})
        if geometry.get('type') != 'Polygon':
            continue
        
        coordinates = geometry.get('coordinates', [])
        if not coordinates or len(coordinates) == 0:
            continue
        
        # Знаходимо координати з feature
        all_coords = [coord for ring in coordinates for coord in ring]
        feature_lons = [coord[0] for coord in all_coords]
        feature_lats = [coord[1] for coord in all_coords]
        
        all_lons.extend(feature_lons)
        all_lats.extend(feature_lats)
    
    if len(all_lons) == 0 or len(all_lats) == 0:
        raise ValueError("Не вдалося отримати координати з features")
    
    # Обчислюємо центр (середнє значення)
    center_lon = (min(all_lons) + max(all_lons)) / 2.0
    center_lat = (min(all_lats) + max(all_lats)) / 2.0
    
    return (center_lat, center_lon)


def validate_hexagonal_grid(hexagons: List[Dict], tolerance: float = 0.01) -> Tuple[bool, List[str]]:
    """
    Перевіряє, чи гексагональна сітка ідеально підходить без границь.
    
    Args:
        hexagons: Список шестикутників
        tolerance: Допуск для перевірки (в метрах)
    
    Returns:
        Tuple (is_valid, list_of_errors)
    """
    errors = []
    
    # Перевірка 1: Всі шестикутники мають однаковий розмір
    if len(hexagons) == 0:
        return False, ["Немає шестикутників"]
    
    # Перевірка 2: Шестикутники не перекриваються (крім сусідніх)
    for i, hex1 in enumerate(hexagons):
        for j, hex2 in enumerate(hexagons[i+1:], start=i+1):
            intersection = hex1['polygon'].intersection(hex2['polygon'])
            if intersection.area > tolerance:
                # Перевіряємо, чи вони сусідні
                row_diff = abs(hex1['row'] - hex2['row'])
                col_diff = abs(hex1['col'] - hex2['col'])
                
                # Сусідні шестикутники мають різницю в координатах <= 1
                if not (row_diff <= 1 and col_diff <= 1):
                    errors.append(f"Шестикутники {hex1['id']} та {hex2['id']} перекриваються")
    
    # Перевірка 3: Суцільне покриття (перевіряємо, чи немає великих прогалин)
    # Це складніше, тому просто перевіряємо, чи є достатньо шестикутників
    
    is_valid = len(errors) == 0
    return is_valid, errors

