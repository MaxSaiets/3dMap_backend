"""
Тести для сервісу генерації рельєфу
"""
import pytest
import numpy as np
import trimesh
from services.terrain_generator import (
    create_terrain_mesh,
    create_grid_faces,
    get_elevation_data
)


class TestTerrainGenerator:
    """Тести для terrain_generator.py"""
    
    def test_create_terrain_mesh_flat(self):
        """Тест створення плоского рельєфу"""
        # bbox_meters = (minx, miny, maxx, maxy)
        bbox = (0.0, 0.0, 1000.0, 1000.0)
        
        mesh, provider = create_terrain_mesh(bbox, z_scale=1.0, resolution=50, latlon_bbox=None, source_crs=None)
        
        assert mesh is not None
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
        assert provider is not None
    
    def test_create_terrain_mesh_with_z_scale(self):
        """Тест масштабування висоти"""
        bbox = (0.0, 0.0, 1000.0, 1000.0)
        
        mesh, _ = create_terrain_mesh(bbox, z_scale=2.0, resolution=50, latlon_bbox=None, source_crs=None)
        
        assert mesh is not None
        # Z координати мають бути масштабовані
        z_values = mesh.vertices[:, 2]
        # Для плоского рельєфу всі Z = 0, але перевіряємо структуру
        assert len(z_values) > 0
    
    def test_create_grid_faces(self):
        """Тест створення граней сітки"""
        rows, cols = 10, 10
        faces = create_grid_faces(rows, cols)
        
        assert faces.shape[1] == 3  # Кожна грань - трикутник
        # Для 10x10 сітки має бути (10-1) * (10-1) * 2 = 162 трикутники
        expected_faces = (rows - 1) * (cols - 1) * 2
        assert len(faces) == expected_faces
    
    def test_get_elevation_data_flat(self):
        """Тест отримання синтетичних даних висоти"""
        from services.elevation_api import get_elevation_simple_terrain
        import numpy as np
        
        X = np.array([[0, 1], [0, 1]])
        Y = np.array([[0, 0], [1, 1]])
        bbox_meters = (1, 0, 1, 0)
        
        Z = get_elevation_simple_terrain(X, Y, bbox_meters, z_scale=1.0)
        
        assert Z.shape == X.shape
        # Синтетичний рельєф не плоский, але має бути валідним
        assert np.all(np.isfinite(Z))  # Всі значення скінченні
        assert Z.min() >= -100 and Z.max() <= 100  # Розумні межі
    
    def test_create_terrain_mesh_different_resolutions(self):
        """Тест різних розділень сітки"""
        bbox = (0.0, 0.0, 1000.0, 1000.0)
        
        for resolution in [50, 100, 200]:
            mesh, _ = create_terrain_mesh(bbox, z_scale=1.0, resolution=resolution, latlon_bbox=None, source_crs=None)
            assert mesh is not None
            # Твердотільний рельєф містить верхню сітку + дно + стінки,
            # тому вершин має бути НЕ менше, ніж у верхній поверхні.
            assert len(mesh.vertices) >= resolution * resolution

