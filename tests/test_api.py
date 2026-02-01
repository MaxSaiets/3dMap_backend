"""
Тести для API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    """Тестовий клієнт FastAPI"""
    return TestClient(app)


class TestAPI:
    """Тести для API endpoints"""
    
    def test_root_endpoint(self, client):
        """Тест кореневого endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_generate_endpoint_valid_request(self, client):
        """Тест endpoint генерації з валідним запитом"""
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
        assert "task_id" in data
        assert "status" in data
    
    def test_generate_endpoint_invalid_bbox(self, client):
        """Тест endpoint генерації з невалідним bbox"""
        request_data = {
            "north": 50.450,  # менше за south
            "south": 50.455,
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
        # Має прийняти запит, але генерація може не вдатися
        assert response.status_code in [200, 422]
    
    def test_status_endpoint_nonexistent_task(self, client):
        """Тест endpoint статусу для неіснуючої задачі"""
        response = client.get("/api/status/nonexistent-task-id")
        assert response.status_code == 404
    
    def test_download_endpoint_nonexistent_task(self, client):
        """Тест endpoint завантаження для неіснуючої задачі"""
        response = client.get("/api/download/nonexistent-task-id")
        assert response.status_code == 404
    
    def test_generate_endpoint_missing_fields(self, client):
        """Тест endpoint генерації з відсутніми полями"""
        request_data = {
            "north": 50.455,
            "south": 50.450,
            # Відсутні інші поля
        }
        
        response = client.post("/api/generate", json=request_data)
        assert response.status_code == 422  # Validation error

