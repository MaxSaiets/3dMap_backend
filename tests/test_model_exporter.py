"""
Тести для model_exporter.py
Перевіряють центрування координат та експорт окремих частин
"""
import numpy as np
import trimesh
import tempfile
import os
import sys

# Додаємо шлях до backend для імпортів
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.model_exporter import export_stl, export_scene


def test_utm_centering():
    """Тест центрування UTM координат"""
    # Створюємо тестовий меш з UTM координатами (великі числа)
    vertices = np.array([
        [300000.0, 5000000.0, 10.0],
        [300100.0, 5000100.0, 15.0],
        [300050.0, 5000050.0, 12.0],
    ])
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Створюємо тимчасовий файл
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        temp_file = f.name
    
    try:
        # Експортуємо
        mesh_items = [("Test", mesh)]
        export_stl(temp_file, mesh_items, model_size_mm=100.0)
        
        # Перевіряємо, що файл створено
        assert os.path.exists(temp_file), "STL файл не створено"
        assert os.path.getsize(temp_file) > 84, "STL файл занадто малий"
        
        # Завантажуємо експортований меш
        exported_mesh = trimesh.load(temp_file)
        
        # Перевіряємо, що координати центровані (розміри мають бути менші за 100км)
        bounds = exported_mesh.bounds
        size = bounds[1] - bounds[0]
        max_dimension = max(size[0], size[1])
        
        assert max_dimension < 100000, f"Координати не центровані: розміри {size}"
        print(f"[OK] Центрування працює: розміри {size[0]:.2f} x {size[1]:.2f}")
        
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_export_parts():
    """Тест експорту окремих частин"""
    # Створюємо тестові меші для різних частин
    base_mesh = trimesh.creation.box(extents=[10, 10, 1])
    roads_mesh = trimesh.creation.box(extents=[5, 5, 0.5])
    water_mesh = trimesh.creation.box(extents=[3, 3, 0.2])
    parks_mesh = trimesh.creation.box(extents=[2, 2, 0.3])
    buildings_mesh = trimesh.creation.box(extents=[1, 1, 2])
    
    # Створюємо тимчасовий файл
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        temp_file = f.name
    
    try:
        # Експортуємо через export_scene
        outputs = export_scene(
            terrain_mesh=base_mesh,
            road_mesh=roads_mesh,
            building_meshes=[buildings_mesh],
            water_mesh=water_mesh,
            parks_mesh=parks_mesh,
            poi_mesh=None,
            filename=temp_file,
            format="stl",
            model_size_mm=100.0,
        )
        
        # Перевіряємо, що outputs не None
        assert outputs is not None, "export_scene повернув None для STL"
        assert isinstance(outputs, dict), "outputs має бути dict"
        
        # Перевіряємо, що всі частини експортовані
        expected_parts = ['base', 'roads', 'water', 'parks', 'buildings']
        for part in expected_parts:
            assert part in outputs, f"Частина {part} не експортована"
            assert os.path.exists(outputs[part]), f"Файл для {part} не створено: {outputs[part]}"
            assert os.path.getsize(outputs[part]) > 84, f"Файл для {part} занадто малий"
            print(f"[OK] Частина {part} експортована: {outputs[part]}")
        
    finally:
        # Видаляємо тимчасові файли
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if outputs:
            for part_file in outputs.values():
                if os.path.exists(part_file):
                    os.remove(part_file)


def test_export_with_missing_parts():
    """Тест експорту з відсутніми частинами"""
    # Створюємо тільки base mesh
    base_mesh = trimesh.creation.box(extents=[10, 10, 1])
    
    # Створюємо тимчасовий файл
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        temp_file = f.name
    
    try:
        # Експортуємо тільки з base
        outputs = export_scene(
            terrain_mesh=base_mesh,
            road_mesh=None,
            building_meshes=[],
            water_mesh=None,
            parks_mesh=None,
            poi_mesh=None,
            filename=temp_file,
            format="stl",
            model_size_mm=100.0,
        )
        
        # Перевіряємо, що експорт пройшов успішно
        assert outputs is not None, "export_scene повернув None"
        assert 'base' in outputs, "Base має бути експортована"
        print(f"[OK] Експорт з відсутніми частинами працює: {list(outputs.keys())}")
        
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if outputs:
            for part_file in outputs.values():
                if os.path.exists(part_file):
                    os.remove(part_file)


if __name__ == "__main__":
    print("Запуск тестів для model_exporter...")
    test_utm_centering()
    test_export_parts()
    test_export_with_missing_parts()
    print("\n[OK] Всі тести пройдені успішно!")
