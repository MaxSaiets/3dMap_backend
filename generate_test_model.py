"""
Скрипт для генерації тестової моделі центру Києва (1км x 1км)
"""
import sys
from pathlib import Path

# Додаємо поточну директорію до шляху
sys.path.insert(0, str(Path(__file__).parent))

from services.data_loader import fetch_city_data
from services.terrain_generator import create_terrain_mesh
from services.model_exporter import export_scene
from services.model_exporter import export_preview_parts_stl
from pathlib import Path

# Центр Києва
KYIV_CENTER_LAT = 50.4501
KYIV_CENTER_LON = 30.5234

# 1 км = приблизно 0.009 градусів на широті
# 1 км = приблизно 0.009 / cos(latitude) градусів на довготі
KM_TO_DEGREES_LAT = 0.009
KM_TO_DEGREES_LON = 0.009 / 0.64  # cos(50°) ≈ 0.64

# Область 1км x 1км
HALF_KM = 0.5
north = KYIV_CENTER_LAT + HALF_KM * KM_TO_DEGREES_LAT
south = KYIV_CENTER_LAT - HALF_KM * KM_TO_DEGREES_LAT
east = KYIV_CENTER_LON + HALF_KM * KM_TO_DEGREES_LON
west = KYIV_CENTER_LON - HALF_KM * KM_TO_DEGREES_LON

print("=" * 80)
print("ГЕНЕРАЦІЯ ТЕСТОВОЇ МОДЕЛІ ЦЕНТРУ КИЄВА (РЕЛЬЄФ ТА ВОДА)")
print("=" * 80)
print(f"Координати:")
print(f"  Північ: {north:.6f}")
print(f"  Південь: {south:.6f}")
print(f"  Схід: {east:.6f}")
print(f"  Захід: {west:.6f}")
print(f"Розмір: ~1км x 1км")
print()
print("РЕЖИМ ТЕСТУВАННЯ: Генерація рельєфу та води (без будівель, доріг)")
print()

# Параметри генерації (тільки для рельєфу)
terrain_enabled = True
terrain_z_scale = 1.5
terrain_base_thickness_mm = 2.0
terrain_resolution = 200
terrarium_zoom = 15
terrain_smoothing_sigma = 0.6
export_format = "3mf"
model_size_mm = 100.0  # 10 см

# Завантажуємо дані OSM тільки для отримання bounds та CRS
print("Завантаження даних OSM (для визначення меж)...")
gdf_buildings, gdf_water, G_roads = fetch_city_data(north, south, east, west)
print(f"Завантажено: {len(gdf_buildings)} будівель, {len(gdf_water)} водних об'єктів (використовується тільки для меж)")

print("Генерація рельєфу...")
terrain_mesh = None
terrain_provider = None
if terrain_enabled:
    # ВАЖЛИВО: рельєф будуємо в UTM, як і OSM-геометрія
    if not gdf_buildings.empty:
        minx, miny, maxx, maxy = gdf_buildings.total_bounds
    else:
        # fallback на дороги
        import osmnx as ox
        gdf_edges = None
        if G_roads is not None:
            if hasattr(G_roads, "total_bounds"):
                gdf_edges = G_roads
            else:
                gdf_edges = ox.graph_to_gdfs(G_roads, nodes=False)
        minx, miny, maxx, maxy = gdf_edges.total_bounds if gdf_edges is not None and not gdf_edges.empty else (-500, -500, 500, 500)
    bbox_meters = (float(minx), float(miny), float(maxx), float(maxy))
    bbox_degrees = (north, south, east, west)
    # scale_factor: meters -> model millimeters
    size_x = float(maxx - minx)
    size_y = float(maxy - miny)
    avg_xy = (size_x + size_y) / 2.0 if (size_x > 0 and size_y > 0) else max(size_x, size_y)
    scale_factor = float(model_size_mm) / float(avg_xy) if avg_xy and avg_xy > 0 else None
    source_crs = None
    try:
        if not gdf_buildings.empty:
            source_crs = gdf_buildings.crs
        elif G_roads is not None and hasattr(G_roads, "crs"):
            source_crs = getattr(G_roads, "crs", None)
        else:
            source_crs = None
    except Exception:
        source_crs = None

    # ТЕСТОВИЙ РЕЖИМ: створюємо тільки рельєф та воду (без будівель, доріг)
    # water depth in meters before scaling
    # ВАЖЛИВО: обчислюємо water_depth_m ПЕРЕД створенням рельєфу, щоб правильно вирізати depression
    water_depth = 2.0  # мм
    has_water = gdf_water is not None and not gdf_water.empty
    water_depth_m = None
    if has_water:
        if scale_factor and scale_factor > 0:
            water_depth_m = float(water_depth) / float(scale_factor)
        else:
            # Fallback: використовуємо приблизну глибину (2мм на моделі = ~0.002м у світі для 100мм моделі)
            water_depth_m = float(water_depth) / 1000.0  # мм -> метри
    
    # Передаємо water_geometries тільки якщо є вода та water_depth_m > 0
    water_geoms_for_terrain = None
    water_depth_for_terrain = 0.0
    if has_water and water_depth_m is not None and water_depth_m > 0:
        water_geoms_for_terrain = list(gdf_water.geometry.values)
        water_depth_for_terrain = float(water_depth_m)

    # Create GlobalCenter
    from services.global_center import GlobalCenter
    # Use center of bbox for GlobalCenter
    center_lat = (north + south) / 2.0
    center_lon = (east + west) / 2.0
    global_center = GlobalCenter(center_lat, center_lon)
    print(f"[DEBUG] Created GlobalCenter: {center_lat}, {center_lon}")

    terrain_mesh, terrain_provider = create_terrain_mesh(
        bbox_meters,
        z_scale=terrain_z_scale,
        resolution=terrain_resolution,
        latlon_bbox=bbox_degrees,
        source_crs=source_crs,
        terrarium_zoom=terrarium_zoom,
        base_thickness=(float(terrain_base_thickness_mm) / float(scale_factor)) if scale_factor else 5.0,
        flatten_buildings=False,
        building_geometries=None,
        flatten_roads=False,
        road_geometries=None,
        smoothing_sigma=float(terrain_smoothing_sigma),
        water_geometries=water_geoms_for_terrain,
        water_depth_m=water_depth_for_terrain,
        subdivide=True,
        subdivide_levels=1,
        global_center=global_center, # Pass global_center
    )
    if terrain_mesh:
        print(f"Рельєф створено: {len(terrain_mesh.vertices)} вершин, {len(terrain_mesh.faces)} граней")
    
    # В тестовому режимі не обробляємо дороги, будівлі, парки, POI, але додаємо воду
    road_mesh = None
    building_meshes = None
    parks_mesh = None
    poi_mesh = None
    
    # Створюємо water mesh для тестового режиму
    water_mesh = None
    if has_water and terrain_provider is not None and water_depth_m is not None and water_depth_m > 0:
        print("Створення води для тестування...")
        from services.water_processor import process_water_surface
        
        # Збільшуємо товщину води для кращої видимості (1.0-2.0мм на моделі)
        max_thickness_mm = min(water_depth * 0.5, 2.0)
        surface_mm = float(max(1.0, min(max_thickness_mm, water_depth * 0.3)))
        thickness_m = float(surface_mm) / float(scale_factor) if scale_factor else (water_depth_m * 0.3)
        water_mesh = process_water_surface(
            gdf_water,
            thickness_m=float(thickness_m),
            depth_meters=float(water_depth_m),
            terrain_provider=terrain_provider,
            global_center=global_center, # Pass global_center
        )
        if water_mesh:
            print(f"Вода створена: {len(water_mesh.vertices)} вершин, {len(water_mesh.faces)} граней")
    
    print("Обробка доріг для тестування (Fix Verification)...")
    road_mesh = None
    if G_roads is not None:
        from services.road_processor import process_roads
        road_mesh = process_roads(
            G_roads,
            terrain_provider=terrain_provider,
            width_multiplier=1.0,
            road_height=0.5,
            road_embed=0.1,
            global_center=global_center, # Pass global_center
            water_geometries=water_geoms_for_terrain, # Pass water for bridge detection
        )
        if road_mesh:
            print(f"Дороги створено: {len(road_mesh.vertices)} вершин.")
        else:
             print("Дороги створено: None/0 вершин. (Можливо обрізані?)")
    else:
        print("Немає даних доріг/графу.")

print("Експорт моделі...")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Експортуємо і 3MF і STL для надійності
output_file_3mf = output_dir / "test_model_kyiv.3mf"
output_file_stl = output_dir / "test_model_kyiv.stl"
output_file_3mf_abs = output_file_3mf.resolve()
output_file_stl_abs = output_file_stl.resolve()

# Експорт 3MF
export_scene(
    terrain_mesh=terrain_mesh,
    road_mesh=road_mesh,
    building_meshes=building_meshes,
    water_mesh=water_mesh,
    parks_mesh=parks_mesh,
    poi_mesh=poi_mesh,
    filename=str(output_file_3mf_abs),
    format="3mf",
    model_size_mm=model_size_mm,
    add_flat_base=not terrain_enabled,
    base_thickness_mm=2.0,
)

# Експорт STL (для надійності)
export_scene(
    terrain_mesh=terrain_mesh,
    road_mesh=road_mesh,
    building_meshes=building_meshes,
    water_mesh=water_mesh,
    parks_mesh=parks_mesh,
    poi_mesh=poi_mesh,
    filename=str(output_file_stl_abs),
    format="stl",
    model_size_mm=model_size_mm,
    add_flat_base=not terrain_enabled,
    base_thickness_mm=2.0,
)

# Експорт STL частин для кольорового прев'ю (тільки рельєф)
print("Експорт STL частин для прев'ю...")
preview_items = []
if terrain_mesh is not None:
    preview_items.append(("Base", terrain_mesh))

if preview_items:
    export_preview_parts_stl(
        output_prefix=str((output_dir / "test_model_kyiv").resolve()),
        mesh_items=preview_items,
        model_size_mm=model_size_mm,
        add_flat_base=not terrain_enabled,
        base_thickness_mm=2.0,
        rotate_to_ground=False,
    )

print()
print("=" * 80)
print("ТЕСТОВА МОДЕЛЬ ГОТОВА! (РЕЛЬЄФ ТА ВОДА)")
print("=" * 80)
print(f"3MF файл: {output_file_3mf_abs}")
print(f"Розмір: {output_file_3mf_abs.stat().st_size:,} байт")
print(f"STL файл: {output_file_stl_abs}")
print(f"Розмір: {output_file_stl_abs.stat().st_size:,} байт")
print()
print("Модель збережена як test_model_kyiv.3mf та test_model_kyiv.stl")
print("Вона буде автоматично завантажена при старті додатка")
print()
print("ПРИМІТКА: Це тестова модель, що містить рельєф та воду (без будівель, доріг)")

