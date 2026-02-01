"""
Тести для сервісу обробки води
"""
import pytest
import geopandas as gpd
import trimesh
from shapely.geometry import Polygon
from services.water_processor import process_water, create_water_depression


class TestWaterProcessor:
    """Тести для water_processor.py"""
    
    def test_process_water_empty(self):
        """Тест обробки порожнього GeoDataFrame"""
        empty_gdf = gpd.GeoDataFrame()
        result = process_water(empty_gdf)
        assert result is None
    
    def test_process_water_with_data(self):
        """Тест обробки водних об'єктів"""
        water = gpd.GeoDataFrame({
            'natural': ['water'],
            'geometry': [
                Polygon([
                    (0, 0),
                    (10, 0),
                    (10, 10),
                    (0, 10)
                ])
            ]
        })
        
        result = process_water(water, depth=2.0, terrain_provider=None)
        
        assert result is not None
        assert isinstance(result, trimesh.Trimesh)
    
    def test_create_water_depression(self):
        """Тест створення западини води"""
        polygon = Polygon([
            (0, 0),
            (10, 0),
            (10, 10),
            (0, 10)
        ])
        
        mesh = create_water_depression(polygon, depth=0.002)  # 2мм в метрах
        
        assert mesh is not None
        assert isinstance(mesh, trimesh.Trimesh)
        # Перевіряємо, що западина має негативну Z координату
        assert mesh.vertices[:, 2].min() < 0
    
    def test_process_water_multipolygon(self):
        """Тест обробки MultiPolygon"""
        from shapely.geometry import MultiPolygon
        
        water = gpd.GeoDataFrame({
            'natural': ['water'],
            'geometry': [
                MultiPolygon([
                    Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
                    Polygon([(10, 10), (15, 10), (15, 15), (10, 15)])
                ])
            ]
        })
        
        result = process_water(water, depth=2.0, terrain_provider=None)
        
        # Має обробити обидва полігони
        assert result is not None

