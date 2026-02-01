"""
Тести для перевірки коректності перетворення координат будівель та їх розміщення на рельєфі
"""
import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from services.global_center import GlobalCenter
from services.terrain_generator import flatten_heightfield_under_buildings
from services.terrain_provider import TerrainProvider
from main import _transform_building_geometries_to_local


class TestBuildingCoordinates:
    """Тести для перетворення координат будівель"""
    
    def test_transform_building_geometries_to_local(self):
        """Тест перетворення геометрій будівель з UTM в локальні координати"""
        # Створюємо глобальний центр (Київ)
        global_center = GlobalCenter(center_lat=50.45, center_lon=30.52)
        
        # Створюємо тестові будівлі в UTM координатах
        # Отримуємо UTM координати для тестових точок
        center_utm_x, center_utm_y = global_center.get_center_utm()
        
        # Створюємо полігон в UTM координатах (наприклад, 100x100 метрів)
        building_polygon_utm = Polygon([
            (center_utm_x - 50, center_utm_y - 50),
            (center_utm_x + 50, center_utm_y - 50),
            (center_utm_x + 50, center_utm_y + 50),
            (center_utm_x - 50, center_utm_y + 50)
        ])
        
        # Створюємо GeoDataFrame з UTM CRS
        gdf_buildings = gpd.GeoDataFrame({
            'building': ['residential'],
            'geometry': [building_polygon_utm]
        }, crs=global_center.utm_crs)
        
        # Перетворюємо в локальні координати
        local_geometries = _transform_building_geometries_to_local(gdf_buildings, global_center)
        
        assert local_geometries is not None
        assert len(local_geometries) == 1
        
        local_geom = local_geometries[0]
        # Перевіряємо, що координати близькі до (0, 0) в локальній системі
        bounds = local_geom.bounds
        assert abs(bounds[0] + 50) < 1.0  # minx приблизно -50
        assert abs(bounds[2] - 50) < 1.0  # maxx приблизно +50
    
    def test_transform_building_geometries_empty(self):
        """Тест обробки порожнього GeoDataFrame"""
        global_center = GlobalCenter(center_lat=50.45, center_lon=30.52)
        empty_gdf = gpd.GeoDataFrame()
        
        result = _transform_building_geometries_to_local(empty_gdf, global_center)
        assert result is None
    
    def test_transform_building_geometries_no_global_center(self):
        """Тест обробки без global_center"""
        building_polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        gdf_buildings = gpd.GeoDataFrame({
            'building': ['residential'],
            'geometry': [building_polygon]
        })
        
        result = _transform_building_geometries_to_local(gdf_buildings, None)
        assert result is None


def _has_rasterio() -> bool:
    """Перевірка наявності rasterio"""
    try:
        import rasterio  # noqa: F401
        return True
    except Exception:
        return False


class TestTerrainFlatteningWithCoordinates:
    """Тести для вирівнювання рельєфу з правильними координатами"""
    
    @pytest.mark.skipif(not _has_rasterio(), reason="rasterio not available")
    def test_flatten_with_local_coordinates(self):
        """Тест вирівнювання рельєфу з локальними координатами будівель"""
        # Створюємо регулярну сітку в локальних координатах
        rows, cols = 50, 50
        x = np.linspace(-100.0, 100.0, cols)
        y = np.linspace(-100.0, 100.0, rows)
        X, Y = np.meshgrid(x, y, indexing="xy")
        
        # Створюємо рельєф з ухилом
        Z = 0.1 * X + 0.05 * Y + 10.0  # Базовий рівень 10м + ухил
        
        # Створюємо будівлю в локальних координатах (центр сітки)
        building_poly = Polygon([
            (-30, -30),
            (30, -30),
            (30, 30),
            (-30, 30)
        ])
        
        # Вирівнюємо рельєф під будівлею
        Z_flattened = flatten_heightfield_under_buildings(
            X=X, Y=Y, Z=Z, 
            building_geometries=[building_poly],
            min_cells=2
        )
        
        # Перевіряємо, що рельєф вирівняний під будівлею
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds
        
        transform = from_bounds(
            float(np.min(X)), float(np.min(Y)), 
            float(np.max(X)), float(np.max(Y)), 
            cols, rows
        )
        mask = rasterize(
            [(building_poly, 1)],
            out_shape=(rows, cols),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=True
        ).astype(bool)
        
        assert int(mask.sum()) > 5  # Має бути достатньо пікселів
        
        # Висоти під будівлею мають бути однаковими (вирівняні)
        heights_under = Z_flattened[mask]
        assert np.all(np.isfinite(heights_under))
        # Різниця між мінімумом і максимумом має бути мінімальною
        height_range = float(np.max(heights_under) - np.min(heights_under))
        assert height_range < 0.01  # Менше 1см різниці
    
    @pytest.mark.skipif(not _has_rasterio(), reason="rasterio not available")
    def test_flatten_preserves_outside(self):
        """Тест, що вирівнювання не змінює рельєф поза будівлею"""
        rows, cols = 40, 40
        x = np.linspace(0.0, 100.0, cols)
        y = np.linspace(0.0, 100.0, rows)
        X, Y = np.meshgrid(x, y, indexing="xy")
        
        # Простий лінійний рельєф
        Z = 0.1 * X + 0.05 * Y
        
        # Будівля в центрі
        building_poly = Polygon([(40, 40), (60, 40), (60, 60), (40, 60)])
        
        Z_flattened = flatten_heightfield_under_buildings(
            X=X, Y=Y, Z=Z,
            building_geometries=[building_poly],
            min_cells=2
        )
        
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds
        
        transform = from_bounds(
            float(np.min(X)), float(np.min(Y)),
            float(np.max(X)), float(np.max(Y)),
            cols, rows
        )
        mask = rasterize(
            [(building_poly, 1)],
            out_shape=(rows, cols),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=True
        ).astype(bool)
        
        # Поза будівлею рельєф має залишитися незмінним
        outside_mask = ~mask
        assert np.allclose(Z_flattened[outside_mask], Z[outside_mask])


class TestBuildingPlacement:
    """Тести для розміщення будівель на рельєфі"""
    
    def test_building_above_ground(self):
        """Тест, що будівля розміщується над землею"""
        from services.building_processor import process_buildings
        
        # Створюємо глобальний центр
        global_center = GlobalCenter(center_lat=50.45, center_lon=30.52)
        
        # Створюємо простий рельєф (TerrainProvider)
        # Створюємо тестовий рельєф з висотами
        rows, cols = 20, 20
        x = np.linspace(-50.0, 50.0, cols)
        y = np.linspace(-50.0, 50.0, rows)
        X, Y = np.meshgrid(x, y, indexing="xy")
        Z = 10.0 + 0.05 * X  # Базовий рівень 10м + легкий ухил
        
        # Створюємо TerrainProvider
        terrain_provider = TerrainProvider(X, Y, Z)
        
        # Створюємо будівлю в локальних координатах
        building_poly = Polygon([
            (-20, -20),
            (20, -20),
            (20, 20),
            (-20, 20)
        ])
        
        # Перетворюємо в UTM для GeoDataFrame
        def to_utm_transform(x, y, z=None):
            x_utm, y_utm = global_center.to_utm(x, y)
            if z is not None:
                return (x_utm, y_utm, z)
            return (x_utm, y_utm)
        
        from shapely.ops import transform
        building_poly_utm = transform(to_utm_transform, building_poly)
        
        gdf_buildings = gpd.GeoDataFrame({
            'building': ['residential'],
            'building:levels': [3],
            'geometry': [building_poly_utm]
        }, crs=global_center.utm_crs)
        
        # Обробляємо будівлі
        building_meshes = process_buildings(
            gdf_buildings,
            min_height=2.0,
            height_multiplier=1.0,
            terrain_provider=terrain_provider,
            foundation_depth=0.5,
            embed_depth=0.0,
            global_center=global_center
        )
        
        if building_meshes and len(building_meshes) > 0:
            mesh = building_meshes[0]
            vertices = mesh.vertices
            
            # Отримуємо висоти рельєфу для нижніх вершин будівлі
            bottom_z = np.min(vertices[:, 2])
            
            # Отримуємо висоти рельєфу під будівлею
            building_xy = vertices[:, :2]
            ground_heights = terrain_provider.get_heights_for_points(building_xy)
            min_ground_height = float(np.min(ground_heights)) if len(ground_heights) > 0 else 0.0
            
            # Будівля має бути над землею (з запасом 0.1м)
            assert bottom_z >= min_ground_height - 0.15, \
                f"Будівля під землею: bottom_z={bottom_z:.2f}, min_ground={min_ground_height:.2f}"
    
    def test_building_safety_margin(self):
        """Тест, що будівля має мінімальний запас над землею"""
        from services.building_processor import process_buildings
        
        global_center = GlobalCenter(center_lat=50.45, center_lon=30.52)
        
        # Створюємо плоский рельєф
        rows, cols = 20, 20
        x = np.linspace(-50.0, 50.0, cols)
        y = np.linspace(-50.0, 50.0, rows)
        X, Y = np.meshgrid(x, y, indexing="xy")
        Z = np.full_like(X, 10.0)  # Плоский рельєф на 10м
        
        terrain_provider = TerrainProvider(X, Y, Z)
        
        # Будівля
        building_poly = Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)])
        
        from shapely.ops import transform
        def to_utm_transform(x, y, z=None):
            x_utm, y_utm = global_center.to_utm(x, y)
            if z is not None:
                return (x_utm, y_utm, z)
            return (x_utm, y_utm)
        
        building_poly_utm = transform(to_utm_transform, building_poly)
        
        gdf_buildings = gpd.GeoDataFrame({
            'building': ['residential'],
            'building:levels': [2],
            'geometry': [building_poly_utm]
        }, crs=global_center.utm_crs)
        
        building_meshes = process_buildings(
            gdf_buildings,
            min_height=2.0,
            height_multiplier=1.0,
            terrain_provider=terrain_provider,
            foundation_depth=0.5,
            embed_depth=0.0,
            global_center=global_center
        )
        
        if building_meshes and len(building_meshes) > 0:
            mesh = building_meshes[0]
            vertices = mesh.vertices
            
            # Нижня точка будівлі
            bottom_z = float(np.min(vertices[:, 2]))
            
            # Очікуваний мінімальний рівень (10м рельєф + 0.1м запас - 0.5м фундамент)
            expected_min = 10.0 + 0.1 - 0.5  # 9.6м
            
            # Перевіряємо, що будівля має запас
            assert bottom_z >= expected_min - 0.1, \
                f"Будівля не має достатнього запасу: bottom_z={bottom_z:.2f}, expected_min={expected_min:.2f}"


class TestIntegrationBuildingTerrain:
    """Інтеграційні тести для повного циклу обробки будівель"""
    
    @pytest.mark.skipif(not _has_rasterio(), reason="rasterio not available")
    @pytest.mark.integration
    def test_full_pipeline_coordinates(self):
        """Тест повного циклу: координати -> вирівнювання -> розміщення"""
        global_center = GlobalCenter(center_lat=50.45, center_lon=30.52)
        
        # 1. Створюємо рельєф
        rows, cols = 30, 30
        x = np.linspace(-50.0, 50.0, cols)
        y = np.linspace(-50.0, 50.0, rows)
        X, Y = np.meshgrid(x, y, indexing="xy")
        Z = 10.0 + 0.1 * X + 0.05 * Y  # Рельєф з ухилом
        
        # 2. Створюємо будівлю в UTM координатах
        center_utm_x, center_utm_y = global_center.get_center_utm()
        building_poly_utm = Polygon([
            (center_utm_x - 20, center_utm_y - 20),
            (center_utm_x + 20, center_utm_y - 20),
            (center_utm_x + 20, center_utm_y + 20),
            (center_utm_x - 20, center_utm_y + 20)
        ])
        
        gdf_buildings = gpd.GeoDataFrame({
            'building': ['residential'],
            'building:levels': [3],
            'geometry': [building_poly_utm]
        }, crs=global_center.utm_crs)
        
        # 3. Перетворюємо координати для вирівнювання
        local_geometries = _transform_building_geometries_to_local(gdf_buildings, global_center)
        assert local_geometries is not None
        assert len(local_geometries) == 1
        
        # 4. Вирівнюємо рельєф
        Z_flattened = flatten_heightfield_under_buildings(
            X=X, Y=Y, Z=Z,
            building_geometries=local_geometries,
            min_cells=2
        )
        
        # 5. Перевіряємо, що рельєф вирівняний
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds
        
        transform = from_bounds(
            float(np.min(X)), float(np.min(Y)),
            float(np.max(X)), float(np.max(Y)),
            cols, rows
        )
        mask = rasterize(
            [(local_geometries[0], 1)],
            out_shape=(rows, cols),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=True
        ).astype(bool)
        
        heights_under = Z_flattened[mask]
        height_range = float(np.max(heights_under) - np.min(heights_under))
        assert height_range < 0.01, f"Рельєф не вирівняний: range={height_range:.4f}"
        
        # 6. Створюємо TerrainProvider з вирівняним рельєфом
        terrain_provider = TerrainProvider(X, Y, Z_flattened)
        
        # 7. Обробляємо будівлі
        from services.building_processor import process_buildings
        
        building_meshes = process_buildings(
            gdf_buildings,
            min_height=2.0,
            height_multiplier=1.0,
            terrain_provider=terrain_provider,
            foundation_depth=0.5,
            embed_depth=0.0,
            global_center=global_center
        )
        
        # 8. Перевіряємо, що будівля над землею
        if building_meshes and len(building_meshes) > 0:
            mesh = building_meshes[0]
            vertices = mesh.vertices
            
            bottom_z = float(np.min(vertices[:, 2]))
            building_xy = vertices[:, :2]
            ground_heights = terrain_provider.get_heights_for_points(building_xy)
            min_ground_height = float(np.min(ground_heights)) if len(ground_heights) > 0 else 0.0
            
            assert bottom_z >= min_ground_height - 0.15, \
                f"Будівля під землею після повного циклу: bottom_z={bottom_z:.2f}, min_ground={min_ground_height:.2f}"

