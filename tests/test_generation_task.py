"""
Тести для управління задачами генерації
"""
import pytest
from services.generation_task import GenerationTask
from pydantic import BaseModel


class MockRequest(BaseModel):
    """Мокований запит"""
    north: float = 50.455
    south: float = 50.450
    east: float = 30.530
    west: float = 30.520


class TestGenerationTask:
    """Тести для generation_task.py"""
    
    def test_task_creation(self):
        """Тест створення задачі"""
        request = MockRequest()
        task = GenerationTask("test-id", request)
        
        assert task.task_id == "test-id"
        assert task.status == "pending"
        assert task.progress == 0
    
    def test_task_update_status(self):
        """Тест оновлення статусу"""
        request = MockRequest()
        task = GenerationTask("test-id", request)
        
        task.update_status("processing", 50, "Обробка...")
        
        assert task.status == "processing"
        assert task.progress == 50
        assert task.message == "Обробка..."
    
    def test_task_complete(self):
        """Тест завершення задачі"""
        request = MockRequest()
        task = GenerationTask("test-id", request)
        
        task.complete("output/test.3mf")
        
        assert task.status == "completed"
        assert task.progress == 100
        assert task.output_file == "output/test.3mf"

    def test_task_set_output(self):
        """Тест збереження вихідних файлів по форматах"""
        request = MockRequest()
        task = GenerationTask("test-id", request)
        task.set_output("3mf", "output/test.3mf")
        task.set_output("stl", "output/test.stl")
        assert task.output_files["3mf"] == "output/test.3mf"
        assert task.output_files["stl"] == "output/test.stl"
    
    def test_task_fail(self):
        """Тест невдачі задачі"""
        request = MockRequest()
        task = GenerationTask("test-id", request)
        
        task.fail("Помилка обробки")
        
        assert task.status == "failed"
        assert task.error == "Помилка обробки"
        assert "Помилка" in task.message

