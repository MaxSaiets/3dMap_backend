# Тести для перевірки коректності роботи з будівлями

Цей файл містить комплексні тести для перевірки коректності:
1. Перетворення координат будівель (UTM -> локальні)
2. Вирівнювання рельєфу під будівлями
3. Розміщення будівель на рельєфі (перевірка, що вони не під землею)
4. Повний інтеграційний цикл

## Встановлення залежностей

```bash
pip install -r requirements-test.txt
```

Або встановіть окремо:
```bash
pip install pytest pytest-cov geopandas numpy shapely trimesh rasterio scipy pyproj
```

## Запуск тестів

### Всі тести для будівель:
```bash
pytest tests/test_building_coordinates.py -v
```

### Конкретний тест:
```bash
pytest tests/test_building_coordinates.py::TestBuildingCoordinates::test_transform_building_geometries_to_local -v
```

### З покриттям коду:
```bash
pytest tests/test_building_coordinates.py --cov=services.building_processor --cov=services.terrain_generator --cov=main -v
```

### Тільки інтеграційні тести:
```bash
pytest tests/test_building_coordinates.py -m integration -v
```

## Структура тестів

### TestBuildingCoordinates
- `test_transform_building_geometries_to_local` - перевірка перетворення координат
- `test_transform_building_geometries_empty` - обробка порожніх даних
- `test_transform_building_geometries_no_global_center` - обробка без global_center

### TestTerrainFlatteningWithCoordinates
- `test_flatten_with_local_coordinates` - вирівнювання рельєфу з локальними координатами
- `test_flatten_preserves_outside` - перевірка, що рельєф поза будівлею не змінюється

### TestBuildingPlacement
- `test_building_above_ground` - перевірка, що будівля над землею
- `test_building_safety_margin` - перевірка мінімального запасу над землею

### TestIntegrationBuildingTerrain
- `test_full_pipeline_coordinates` - повний інтеграційний тест

## Що перевіряють тести

1. **Перетворення координат**: 
   - Будівлі в UTM координатах правильно перетворюються в локальні
   - Порожні дані обробляються коректно
   - Відсутність global_center обробляється gracefully

2. **Вирівнювання рельєфу**:
   - Рельєф під будівлею вирівнюється до однакового рівня (медіана)
   - Рельєф поза будівлею залишається незмінним
   - Використовуються правильні локальні координати

3. **Розміщення будівель**:
   - Будівлі завжди розміщуються над землею
   - Є мінімальний запас 0.1м над землею
   - Правильно враховується фундамент та embed_depth

4. **Інтеграційний тест**:
   - Повний цикл: координати -> вирівнювання -> розміщення
   - Всі компоненти працюють разом коректно

## Очікувані результати

Всі тести мають проходити успішно. Якщо тест падає:

1. **test_transform_building_geometries_to_local** - перевірте, що `_transform_building_geometries_to_local` правильно перетворює координати
2. **test_flatten_with_local_coordinates** - перевірте, що `flatten_heightfield_under_buildings` використовує медіану та правильно вирівнює рельєф
3. **test_building_above_ground** - перевірте, що `process_buildings` правильно розміщує будівлі над землею з запасом
4. **test_full_pipeline_coordinates** - перевірте весь цикл обробки

## Примітки

- Деякі тести потребують `rasterio` - вони автоматично пропускаються, якщо бібліотека не встановлена
- Інтеграційні тести позначені маркером `@pytest.mark.integration`
- Тести використовують тестові дані (Київ, невелика область)

