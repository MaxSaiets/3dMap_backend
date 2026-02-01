# Тести Backend

## Запуск тестів

```bash
# Встановити тестові залежності
pip install -r requirements-test.txt

# Запустити всі тести
pytest

# Запустити з покриттям коду
pytest --cov=services --cov-report=html

# Запустити конкретний тест
pytest tests/test_data_loader.py

# Запустити з виводом
pytest -v -s
```

## Структура тестів

- `test_data_loader.py` - Тести завантаження OSM даних
- `test_road_processor.py` - Тести обробки доріг
- `test_building_processor.py` - Тести обробки будівель
- `test_water_processor.py` - Тести обробки води
- `test_terrain_generator.py` - Тести генерації рельєфу
- `test_model_exporter.py` - Тести експорту моделей
- `test_api.py` - Тести API endpoints
- `test_generation_task.py` - Тести управління задачами

## Маркери тестів

- `@pytest.mark.unit` - Unit тести
- `@pytest.mark.integration` - Інтеграційні тести
- `@pytest.mark.slow` - Повільні тести

## Примітки

Деякі тести вимагають інтернет-з'єднання для завантаження OSM даних. Для тестів без інтернету використовуються моки.

