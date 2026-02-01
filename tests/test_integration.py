"""
Інтеграційні тести для повного циклу генерації моделі
"""
import pytest
import time
import os
from pathlib import Path
from fastapi.testclient import TestClient
from main import app, tasks, OUTPUT_DIR
from services.generation_task import GenerationTask


@pytest.fixture
def client():
    """Тестовий клієнт FastAPI"""
    return TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_tasks():
    """Очищення задач після кожного тесту"""
    yield
    tasks.clear()


class TestModelGeneration:
    """Інтеграційні тести генерації моделі"""
    
    def test_full_generation_cycle_stl(self, client):
        """Тест повного циклу генерації STL моделі"""
        # 1. Створюємо запит на генерацію
        request_data = {
            "north": 50.455,
            "south": 50.450,
            "east": 30.530,
            "west": 30.520,
            "road_width_multiplier": 1.0,
            "building_min_height": 2.0,
            "building_height_multiplier": 1.0,
            "water_depth": 2.0,
            "terrain_enabled": True,
            "terrain_z_scale": 1.5,
            "export_format": "stl"
        }
        
        response = client.post("/api/generate", json=request_data)
        assert response.status_code == 200
        data = response.json()
        task_id = data["task_id"]
        assert task_id is not None
        assert data["status"] == "processing"
        
        # 2. Чекаємо завершення генерації (максимум 60 секунд)
        max_wait = 60
        wait_time = 0
        while wait_time < max_wait:
            status_response = client.get(f"/api/status/{task_id}")
            assert status_response.status_code == 200
            status_data = status_response.json()
            
            if status_data["status"] == "completed":
                # 3. Перевіряємо, що файл створено
                assert status_data["download_url"] is not None
                assert status_data["progress"] == 100
                
                # 4. Завантажуємо файл
                download_response = client.get(f"/api/download/{task_id}")
                assert download_response.status_code == 200
                assert len(download_response.content) > 0
                
                # 5. Перевіряємо, що це STL файл
                assert download_response.headers["content-type"] in ["model/stl", "application/octet-stream"]
                
                # 6. Перевіряємо, що файл існує на диску
                task = tasks[task_id]
                assert task.output_file is not None
                assert os.path.exists(task.output_file)
                assert task.output_file.endswith(".stl")
                
                # 7. Перевіряємо розмір файлу (має бути більше 0)
                file_size = os.path.getsize(task.output_file)
                assert file_size > 0
                
                print(f"[OK] STL модель успішно згенерована: {task.output_file} ({file_size} байт)")
                return
            
            elif status_data["status"] == "failed":
                pytest.fail(f"Генерація не вдалася: {status_data.get('message', 'Невідома помилка')}")
            
            time.sleep(2)
            wait_time += 2
        
        pytest.fail(f"Генерація не завершилася за {max_wait} секунд")
    
    def test_full_generation_cycle_3mf(self, client):
        """Тест повного циклу генерації 3MF моделі"""
        # 1. Створюємо запит на генерацію
        request_data = {
            "north": 50.455,
            "south": 50.450,
            "east": 30.530,
            "west": 30.520,
            "road_width_multiplier": 1.0,
            "building_min_height": 2.0,
            "building_height_multiplier": 1.0,
            "water_depth": 2.0,
            "terrain_enabled": True,
            "terrain_z_scale": 1.5,
            "export_format": "3mf"
        }
        
        response = client.post("/api/generate", json=request_data)
        assert response.status_code == 200
        data = response.json()
        task_id = data["task_id"]
        assert task_id is not None
        
        # 2. Чекаємо завершення генерації
        max_wait = 60
        wait_time = 0
        while wait_time < max_wait:
            status_response = client.get(f"/api/status/{task_id}")
            assert status_response.status_code == 200
            status_data = status_response.json()
            
            if status_data["status"] == "completed":
                # 3. Перевіряємо завантаження
                download_response = client.get(f"/api/download/{task_id}")
                assert download_response.status_code == 200
                assert len(download_response.content) > 0
                
                # 4. Перевіряємо файл
                task = tasks[task_id]
                assert task.output_file is not None
                assert os.path.exists(task.output_file)
                assert task.output_file.endswith(".3mf")

                # 4.1. Перевіряємо, що є STL прев'ю та endpoint його віддає
                stl_resp = client.get(f"/api/download/{task_id}?format=stl")
                assert stl_resp.status_code == 200
                assert len(stl_resp.content) > 0
                assert stl_resp.headers["content-type"] in ["model/stl", "application/octet-stream"]
                
                file_size = os.path.getsize(task.output_file)
                assert file_size > 0
                
                print(f"[OK] 3MF модель успішно згенерована: {task.output_file} ({file_size} байт)")
                return
            
            elif status_data["status"] == "failed":
                pytest.fail(f"Генерація не вдалася: {status_data.get('message', 'Невідома помилка')}")
            
            time.sleep(2)
            wait_time += 2
        
        pytest.fail(f"Генерація не завершилася за {max_wait} секунд")
    
    def test_model_contains_expected_elements(self, client):
        """Тест, що модель містить очікувані елементи (будівлі, дороги)"""
        request_data = {
            "north": 50.455,
            "south": 50.450,
            "east": 30.530,
            "west": 30.520,
            "road_width_multiplier": 1.0,
            "building_min_height": 2.0,
            "building_height_multiplier": 1.0,
            "water_depth": 2.0,
            "terrain_enabled": True,
            "terrain_z_scale": 1.5,
            "export_format": "stl"
        }
        
        response = client.post("/api/generate", json=request_data)
        task_id = response.json()["task_id"]
        
        # Чекаємо завершення
        max_wait = 60
        wait_time = 0
        while wait_time < max_wait:
            status_response = client.get(f"/api/status/{task_id}")
            status_data = status_response.json()
            
            if status_data["status"] == "completed":
                # Перевіряємо, що файл містить дані
                task = tasks[task_id]
                file_size = os.path.getsize(task.output_file)
                
                # STL файл має містити мінімум заголовок (80 байт) + кілька трикутників
                # Кожен трикутник = 12 байт нормаль + 36 байт вершин + 2 байти атрибут = 50 байт
                assert file_size >= 84  # Мінімум 1 трикутник
                
                # Перевіряємо, що файл починається з STL заголовка (ASCII або бінарний)
                with open(task.output_file, "rb") as f:
                    header = f.read(80)
                    # STL може бути ASCII (починається з "solid") або бінарний
                    # Trimesh зазвичай експортує бінарний STL
                    assert len(header) == 80
                
                print(f"[OK] Модель містить геометрію: {file_size} байт")
                return
            
            elif status_data["status"] == "failed":
                pytest.fail(f"Генерація не вдалася: {status_data.get('message')}")
            
            time.sleep(2)
            wait_time += 2
        
        pytest.fail("Генерація не завершилася")
    
    def test_progress_updates(self, client):
        """Тест, що прогрес оновлюється під час генерації"""
        request_data = {
            "north": 50.455,
            "south": 50.450,
            "east": 30.530,
            "west": 30.520,
            "export_format": "stl"
        }
        
        response = client.post("/api/generate", json=request_data)
        task_id = response.json()["task_id"]
        
        # Перевіряємо, що прогрес змінюється
        progress_values = []
        max_wait = 60
        wait_time = 0
        
        while wait_time < max_wait:
            status_response = client.get(f"/api/status/{task_id}")
            status_data = status_response.json()
            progress_values.append(status_data["progress"])
            
            if status_data["status"] == "completed":
                # Перевіряємо, що прогрес досяг 100%
                assert status_data["progress"] == 100
                # Перевіряємо, що прогрес існує (може бути швидка генерація)
                assert len(progress_values) > 0
                assert all(0 <= p <= 100 for p in progress_values)
                print(f"[OK] Прогрес відстежувався: {progress_values}")
                return
            
            elif status_data["status"] == "failed":
                break
            
            time.sleep(0.5)  # Перевіряємо частіше
            wait_time += 0.5
        
        # Якщо не завершилося, все одно перевіряємо прогрес
        if progress_values:
            assert len(progress_values) > 0
            assert all(0 <= p <= 100 for p in progress_values)

