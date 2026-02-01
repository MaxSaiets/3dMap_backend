"""
Green areas (parks/forests/grass) processor.

Creates a thin embossed mesh that is draped onto terrain:
new_z = ground_z + old_z - embed

This makes parks/green areas stand out visually and be printable (has thickness).
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon, box, Point
from shapely.ops import transform, unary_union

from services.terrain_provider import TerrainProvider
from services.global_center import GlobalCenter


def _create_high_res_mesh(poly: Polygon, height_m: float, target_edge_len_m: float) -> Optional[trimesh.Trimesh]:
    """
    Створює меш з UNIFORM тріангуляцією (remeshed) використовуючи Steiner points.
    Виправляє проблему діагональних смуг, створюючи рівномірні трикутники.
    
    Підхід:
    1. Resample Boundary - додає точки на контур для точного накладання на рельєф
    2. Internal Grid - генерує рівномірну сітку всередині полігону
    3. Delaunay Triangulation - створює рівномірні трикутники
    4. Extrude - витягує в 3D з боковими стінками
    
    Args:
        poly: Вхідний полігон
        height_m: Висота екструзії
        target_edge_len_m: Цільова довжина ребра в метрах (максимальна)
    
    Returns:
        Trimesh об'єкт з високою деталізацією та рівномірною топологією
    """
    try:
        if poly is None or poly.is_empty:
            return None
        
        if target_edge_len_m <= 0:
            target_edge_len_m = 3.0
        
        # 1. RESAMPLE BOUNDARY (Виправляє нерівні краї)
        # Розбиваємо контур на дрібні відрізки для точного накладання на рельєф
        boundary_coords = []
        
        # Обробляємо зовнішній контур
        exterior_pts = np.array(poly.exterior.coords[:-1])  # Без останньої точки (дублікат першої)
        for i in range(len(exterior_pts)):
            p1 = exterior_pts[i]
            p2 = exterior_pts[(i + 1) % len(exterior_pts)]
            dist = np.linalg.norm(p2 - p1)
            
            if dist > target_edge_len_m:
                # Додаємо проміжні точки
                num_segments = int(np.ceil(dist / target_edge_len_m))
                t = np.linspace(0, 1, num_segments + 1)[:-1]  # Без останньої
                for val in t:
                    boundary_coords.append(p1 + (p2 - p1) * val)
            else:
                boundary_coords.append(p1)
        
        # Обробляємо внутрішні отвори (якщо є)
        for interior in poly.interiors:
            interior_pts = np.array(interior.coords[:-1])
            for i in range(len(interior_pts)):
                p1 = interior_pts[i]
                p2 = interior_pts[(i + 1) % len(interior_pts)]
                dist = np.linalg.norm(p2 - p1)
                
                if dist > target_edge_len_m:
                    num_segments = int(np.ceil(dist / target_edge_len_m))
                    t = np.linspace(0, 1, num_segments + 1)[:-1]
                    for val in t:
                        boundary_coords.append(p1 + (p2 - p1) * val)
                else:
                    boundary_coords.append(p1)
        
        # 2. GENERATE INTERNAL GRID (Виправляє діагональні смуги)
        # Створюємо рівномірну сітку всередині полігону
        minx, miny, maxx, maxy = poly.bounds
        
        # Створюємо сітку з кроком target_edge_len_m
        x_range = np.arange(minx, maxx, target_edge_len_m)
        y_range = np.arange(miny, maxy, target_edge_len_m)
        
        if len(x_range) == 0:
            x_range = np.array([minx, maxx])
        if len(y_range) == 0:
            y_range = np.array([miny, maxy])
        
        xx, yy = np.meshgrid(x_range, y_range)
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
        
        # Фільтруємо точки всередині полігону (використовуємо prepared geometry для швидкості)
        try:
            from shapely.prepared import prep
            prep_poly = prep(poly)
            valid_points = []
            for pt in grid_points:
                if prep_poly.contains(Point(pt[0], pt[1])):
                    valid_points.append(pt)
        except ImportError:
            # Fallback без prepared geometry
            valid_points = []
            for pt in grid_points:
                if poly.contains(Point(pt[0], pt[1])):
                    valid_points.append(pt)
        
        # Об'єднуємо контурні та внутрішні точки
        if len(boundary_coords) == 0:
            # Fallback до простого extrude, якщо не вдалося створити точки
            return trimesh.creation.extrude_polygon(poly, height=float(height_m))
        
        all_points = np.array(boundary_coords + valid_points)
        
        # Видаляємо дублікати (точки, що дуже близькі одна до одної)
        # Використовуємо простий підхід: групуємо точки за округленими координатами
        tolerance = target_edge_len_m * 0.1  # 10% від цільової довжини
        unique_points = []
        seen = set()
        for pt in all_points:
            key = (round(pt[0] / tolerance), round(pt[1] / tolerance))
            if key not in seen:
                seen.add(key)
                unique_points.append(pt)
        
        if len(unique_points) < 3:
            # Fallback
            return trimesh.creation.extrude_polygon(poly, height=float(height_m))
        
        all_points = np.array(unique_points)
        
        # 3. DELAUNAY TRIANGULATION (Створює рівномірні трикутники)
        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(all_points)
            
            # Delaunay створює Convex Hull, тому треба відкинути трикутники ПОЗА полігоном
            faces = tri.simplices
            vertices = tri.points
            
            # Використовуємо prepared geometry для швидкої перевірки
            try:
                prep_poly = prep(poly)
                contains_check = lambda pt: prep_poly.contains(Point(pt[0], pt[1]))
            except:
                contains_check = lambda pt: poly.contains(Point(pt[0], pt[1]))
            
            final_faces = []
            for face in faces:
                # Рахуємо центр трикутника
                centroid = np.mean(vertices[face], axis=0)
                if contains_check(centroid):
                    final_faces.append(face)
            
            if len(final_faces) == 0:
                # Fallback
                return trimesh.creation.extrude_polygon(poly, height=float(height_m))
            
            # Створюємо плоский 2D меш
            mesh_2d = trimesh.Trimesh(vertices=vertices, faces=np.array(final_faces))
            
        except ImportError:
            # Якщо scipy недоступний, використовуємо простий extrude з subdivision
            print("[WARN] scipy недоступний, використовується простий extrude з subdivision")
            mesh = trimesh.creation.extrude_polygon(poly, height=float(height_m))
            if mesh is None:
                return None
            
            # Адаптивний subdivision як fallback
            minx, miny, maxx, maxy = poly.bounds
            max_dim = max(maxx - minx, maxy - miny)
            needed_subdivisions = max(1, min(int(np.ceil(np.log2(max_dim / target_edge_len_m))), 6))
            
            for _ in range(needed_subdivisions):
                if len(mesh.vertices) > 150000:
                    break
                try:
                    mesh = mesh.subdivide()
                except Exception:
                    break
            return mesh
        
        # 4. EXTRUSION (Витягуємо в 3D з боковими стінками)
        # Створюємо нижні та верхні вершини
        n_verts = len(mesh_2d.vertices)
        v_bottom = np.column_stack((mesh_2d.vertices, np.zeros(n_verts)))
        v_top = np.column_stack((mesh_2d.vertices, np.full(n_verts, float(height_m))))
        
        # Об'єднуємо вершини
        vertices_3d = np.vstack((v_bottom, v_top))
        
        # Створюємо грані: нижня поверхня (flipped), верхня поверхня
        f_bottom = np.fliplr(mesh_2d.faces)  # Перевертаємо для правильної нормалі
        f_top = mesh_2d.faces + n_verts
        
        # Створюємо бокові стінки
        # Знаходимо boundary edges (ребра, що належать тільки одному трикутнику)
        edges = mesh_2d.edges
        edge_count = {}
        for face in mesh_2d.faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # Boundary edges - ті, що зустрічаються тільки один раз
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        
        # Створюємо бокові грані для кожного boundary edge
        side_faces = []
        for edge in boundary_edges:
            v1_bottom, v2_bottom = edge[0], edge[1]
            v1_top = v1_bottom + n_verts
            v2_top = v2_bottom + n_verts
            
            # Два трикутники для бокової грані (квад перетворюємо в 2 трикутники)
            side_faces.append([v1_bottom, v2_bottom, v1_top])
            side_faces.append([v2_bottom, v2_top, v1_top])
        
        # Об'єднуємо всі грані
        all_faces = np.vstack([
            f_bottom,
            f_top,
            np.array(side_faces) if side_faces else np.empty((0, 3), dtype=int)
        ])
        
        # Створюємо 3D меш
        mesh_3d = trimesh.Trimesh(vertices=vertices_3d, faces=all_faces)
        
        return mesh_3d
        
    except Exception as e:
        print(f"[WARN] Помилка створення High-Res мешу з Delaunay: {e}")
        import traceback
        traceback.print_exc()
        # Fallback до простого extrude
        try:
            return trimesh.creation.extrude_polygon(poly, height=float(height_m))
        except Exception:
            return None


def process_green_areas(
    gdf_green: gpd.GeoDataFrame,
    height_m: float,
    embed_m: float,
    terrain_provider: Optional[TerrainProvider] = None,
    global_center: Optional[GlobalCenter] = None,  # UTM -> local
    scale_factor: Optional[float] = None,  # model_mm / world_m
    min_feature_mm: float = 0.8,
    simplify_mm: float = 0.4,
    # --- НОВИЙ АРГУМЕНТ: Полігони доріг для вирізання ---
    road_polygons: Optional[object] = None,  # Shapely Polygon/MultiPolygon об'єднаних доріг (в локальних координатах)
) -> Optional[trimesh.Trimesh]:
    if gdf_green is None or gdf_green.empty:
        return None

    # --- Coordinate Transform Block ---
    if global_center is not None:
        try:
            def to_local_transform(x, y, z=None):
                x_local, y_local = global_center.to_local(x, y)
                if z is not None:
                    return (x_local, y_local, z)
                return (x_local, y_local)

            gdf_local = gdf_green.copy()
            gdf_local["geometry"] = gdf_local["geometry"].apply(
                lambda geom: transform(to_local_transform, geom) if geom is not None and not geom.is_empty else geom
            )
            gdf_green = gdf_local
        except Exception:
            pass

    # --- Road Mask Preparation (Підготовка маски доріг для вирізання) ---
    # Переконуємось, що полігон доріг теж в локальних координатах
    road_mask = road_polygons
    if road_mask is not None:
        # Перевіряємо, чи маска не порожня
        try:
            if getattr(road_mask, "is_empty", False):
                road_mask = None
        except Exception:
            pass
        
        # Перетворення координат (якщо потрібно)
        if road_mask is not None and global_center is not None:
            # Перевірка: якщо координати доріг виглядають як UTM (великі числа), а ми вже в local
            try:
                bounds = road_mask.bounds
                sample_x = bounds[0]
                # Евристика: якщо координати > 100000, це UTM
                if abs(sample_x) > 100000:
                    def to_local_transform(x, y, z=None):
                        x_local, y_local = global_center.to_local(x, y)
                        if z is not None:
                            return (x_local, y_local, z)
                        return (x_local, y_local)
                    road_mask = transform(to_local_transform, road_mask)
                    # Перевіряємо валідність після перетворення
                    if road_mask is None or getattr(road_mask, "is_empty", False):
                        road_mask = None
            except Exception as e:
                print(f"[WARN] Помилка перетворення road_polygons в локальні координати: {e}")
                # Якщо не вдалося перетворити, не використовуємо маску
                road_mask = None

    # --- Clipping Block ---
    clip_box = None
    if terrain_provider is not None:
        try:
            min_x, max_x, min_y, max_y = terrain_provider.get_bounds()
            clip_box = box(min_x, min_y, max_x, max_y)
        except Exception:
            clip_box = None

    # --- Parameters Calculation ---
    simplify_tol_m = 0.5
    min_width_m = None
    target_edge_len_m = 3.0  # Базове значення
    
    if scale_factor is not None and float(scale_factor) > 0:
        try:
            simplify_tol_m = max(0.05, float(simplify_mm) / float(scale_factor))
        except Exception:
            pass
        try:
            min_width_m = max(0.0, float(min_feature_mm) / float(scale_factor))
        except Exception:
            pass
        
        # Для Low Poly текстури нам потрібна сітка ~1.5 - 2 мм на моделі
        # Це забезпечить достатньо вершин для "піків"
        try:
            target_edge_len_m = 2.0 / float(scale_factor)  # 2mm на моделі
            # Обмежуємо, щоб не повісити систему на великих картах
            target_edge_len_m = max(1.5, min(target_edge_len_m, 10.0))
        except Exception:
            pass

    # --- Polygon Cleaning & Collection ---
    polys: list[Polygon] = []

    def _iter_polys(g):
        if g is None or getattr(g, "is_empty", False):
            return []
        if isinstance(g, Polygon):
            return [g]
        if isinstance(g, MultiPolygon):
            return list(g.geoms)
        if hasattr(g, "geoms"):
            return [gg for gg in g.geoms if isinstance(gg, Polygon)]
        return []

    for _, row in gdf_green.iterrows():
        geom = getattr(row, "geometry", None)
        if geom is None or getattr(geom, "is_empty", False):
            continue
        try:
            if not geom.is_valid:
                geom = geom.buffer(0)
        except Exception:
            pass
        if geom is None or getattr(geom, "is_empty", False):
            continue

        if clip_box is not None:
            try:
                geom = geom.intersection(clip_box)
            except Exception:
                continue
            if geom is None or getattr(geom, "is_empty", False):
                continue

        # --- ROAD CLIPPING (ВИРІЗАННЯ ДОРІГ) ---
        # Це найважливіший момент: віднімаємо дороги від зеленої зони
        # Це запобігає z-fighting та перетину між дорогами та парками
        if road_mask is not None:
            try:
                # ОПТИМІЗАЦІЯ: Обрізаємо road_mask по bounds поточного парку перед вирізанням
                # Це значно прискорює Boolean операцію для великих MultiPolygon доріг
                geom_bounds = geom.bounds
                road_mask_clipped = road_mask
                
                # Створюємо bounding box для парку з невеликим padding (для безпеки)
                padding = 10.0  # 10 метрів padding для уникнення проблем на краях
                clip_box_geom = box(
                    geom_bounds[0] - padding,
                    geom_bounds[1] - padding,
                    geom_bounds[2] + padding,
                    geom_bounds[3] + padding
                )
                
                # Обрізаємо road_mask по bounds парку
                try:
                    road_mask_clipped = road_mask.intersection(clip_box_geom)
                    if road_mask_clipped is None or getattr(road_mask_clipped, "is_empty", False):
                        # Якщо дороги не перетинаються з цим парком, пропускаємо вирізання
                        pass
                    else:
                        # Вирізаємо дороги з парку (Boolean Difference)
                        geom = geom.difference(road_mask_clipped)
                except Exception:
                    # Якщо intersection не вдався, пробуємо без обрізання (повільніше, але надійніше)
                    geom = geom.difference(road_mask)
                    
            except Exception as e:
                print(f"[WARN] Помилка вирізання доріг із парку: {e}")
                # Якщо помилка, залишаємо як є (краще мати парк, ніж нічого)
                pass
            
            # Перевіряємо, чи геометрія не стала порожньою після вирізання
            if geom is None or getattr(geom, "is_empty", False):
                continue

        for poly in _iter_polys(geom):
            if poly is None or getattr(poly, "is_empty", False):
                continue
            try:
                if not poly.is_valid:
                    poly = poly.buffer(0)
            except Exception:
                pass
            if poly is None or getattr(poly, "is_empty", False):
                continue

            # Clean tiny artifacts after clipping (особливо після вирізання доріг)
            try:
                # Після вирізання доріг можуть залишитися дуже маленькі шматочки
                # Видаляємо їх (менше 10м² замість 100м², бо це артефакти від вирізання)
                if float(getattr(poly, "area", 0.0) or 0.0) < 10.0:
                    continue
            except Exception:
                pass
            polys.append(poly)

    if not polys:
        return None

    # Union overlapping parks (після вирізання доріг)
    # ВАЖЛИВО: Union робимо ПІСЛЯ вирізання доріг, щоб уникнути проблем з об'єднанням
    try:
        merged = unary_union(polys)
        polys = []
        for p in _iter_polys(merged):
            if p is not None and not p.is_empty:
                # Додаткова перевірка площі після union (можуть з'явитися нові дрібні шматочки)
                try:
                    if float(getattr(p, "area", 0.0) or 0.0) < 10.0:
                        continue
                except Exception:
                    pass
                polys.append(p)
    except Exception:
        pass

    # Filter & Simplify
    filtered: list[Polygon] = []
    for poly in polys:
        if poly is None or poly.is_empty:
            continue
        try:
            poly = poly.simplify(float(simplify_tol_m), preserve_topology=True)
        except Exception:
            pass
        if poly is None or poly.is_empty:
            continue

        if min_width_m is not None and float(min_width_m) > 0:
            # Simple check via area/perimeter ratio
            try:
                per = float(getattr(poly, "length", 0.0) or 0.0)
                area = float(getattr(poly, "area", 0.0) or 0.0)
                if per > 0 and area > 0:
                    equiv_width = float((2.0 * area) / per)
                    if equiv_width < float(min_width_m):
                        continue
            except Exception:
                pass
        filtered.append(poly)

    if not filtered:
        return None

    # --- MESH GENERATION ---
    meshes: list[trimesh.Trimesh] = []
    for poly in filtered:
        try:
            # 1. Створення високодеталізованого мешу
            mesh = _create_high_res_mesh(poly, float(height_m), target_edge_len_m)

            if mesh is None or len(mesh.vertices) == 0:
                continue

            # 2. Накладання на рельєф (Draping)
            if terrain_provider is not None:
                v = mesh.vertices.copy()
                old_z = v[:, 2].copy()
                ground_heights = terrain_provider.get_surface_heights_for_points(v[:, :2])
                
                z_min = float(np.min(old_z))
                z_max = float(np.max(old_z))
                z_range = z_max - z_min
                
                relative_height = np.zeros_like(old_z)
                if z_range > 1e-6:
                    relative_height = (old_z - z_min) / z_range
                
                new_z = ground_heights - float(embed_m) + relative_height * float(height_m)
                
                # Локальна корекція (щоб не провалювалось)
                safety_margin = 0.01 
                min_allowed_z = ground_heights + safety_margin - float(embed_m)
                new_z = np.maximum(new_z, min_allowed_z)
                
                # ВИПРАВЛЕННЯ Z-FIGHTING: Забезпечуємо, що парки трохи нижчі за дороги на стику
                # Дороги мають min_top_clearance = road_height * 0.02 (2% від висоти)
                # Парки мають бути нижче, щоб уникнути z-fighting
                # Додаємо невеликий offset для верхніх вершин парку (0.5% від висоти парку)
                z_fighting_offset = float(height_m) * 0.005  # 0.5% від висоти парку
                top_vertices_mask = relative_height > 0.9  # Верхні 10% вершин
                if np.any(top_vertices_mask):
                    # Трохи опускаємо верхні вершини парку для уникнення z-fighting з дорогами
                    new_z[top_vertices_mask] = new_z[top_vertices_mask] - z_fighting_offset
                
                v[:, 2] = new_z
                mesh.vertices = v
        
            # 3. Додавання текстури з маскою країв
            mesh = _add_strong_faceted_texture(mesh, height_m, scale_factor, original_polygon=poly)

            if len(mesh.faces) > 0:
                meshes.append(mesh)

        except Exception as e:
            print(f"[WARN] Помилка обробки полігону: {e}")
            continue

    if not meshes:
        return None

    try:
        return trimesh.util.concatenate(meshes)
    except Exception:
        return meshes[0]


def _add_strong_faceted_texture(
    mesh: trimesh.Trimesh, 
    height_m: float, 
    scale_factor: Optional[float] = None,
    original_polygon: Optional[Polygon] = None
) -> trimesh.Trimesh:
    """
    Додає Low Poly текстуру з урахуванням маски країв (Boundary Masking).
    
    Вершини на краях полігону отримують weight = 0 (чисті краї для стикування з дорогами),
    вершини в центрі отримують weight = 1 (повний шум для Low Poly ефекту).
    
    Args:
        mesh: Меш після накладання на рельєф
        height_m: Висота мешу в метрах (для fallback розрахунків)
        scale_factor: Масштаб моделі (model_mm / world_m) для print-aware розрахунків
        original_polygon: Оригінальний полігон для обчислення відстані до краю
    
    Returns:
        Меш з доданою Low Poly текстурою
    """
    try:
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return mesh

        # Знаходимо "дах" (верхні вершини)
        if mesh.vertex_normals is None or len(mesh.vertex_normals) != len(mesh.vertices):
            mesh.fix_normals()
        
        up_facing = mesh.vertex_normals[:, 2] > 0.5
        
        # Fallback якщо нормалі погані
        if not np.any(up_facing):
            z_values = mesh.vertices[:, 2]
            max_z = float(np.max(z_values))
            min_z = float(np.min(z_values))
            z_range = max_z - min_z
            if z_range < 0.01:
                return mesh
            threshold = min_z + z_range * 0.80
            up_facing = z_values > threshold

        if not np.any(up_facing):
            return mesh

        top_indices = np.where(up_facing)[0]
        top_vertices_xy = mesh.vertices[top_indices, :2]

        # Параметри текстури
        target_noise_mm = 0.5  # Бажана висота шуму на моделі (0.5мм)
        texture_amplitude = min(height_m * 0.4, 3.0)  # default fallback
        fade_distance_m = 2.0  # Дистанція від краю, де починається шум (в метрах)

        if scale_factor is not None and float(scale_factor) > 0:
            texture_amplitude = target_noise_mm / float(scale_factor)
            texture_amplitude = min(texture_amplitude, 3.0)
            # fade distance теж залежить від масштабу
            # ~2-3мм на моделі для плавного переходу
            fade_distance_m = max(2.0, 3.0 / float(scale_factor))

        # --- Boundary Masking (Маска країв) ---
        noise_weights = np.ones(len(top_indices), dtype=float)
        
        if original_polygon is not None and not original_polygon.is_empty:
            try:
                # Shapely boundary для розрахунку відстані
                boundary = original_polygon.boundary
                
                # Для кожної верхньої вершини обчислюємо відстань до краю
                # Це може бути повільно для 50k+ точок, але необхідно для якості
                for i, (x, y) in enumerate(top_vertices_xy):
                    pt = Point(x, y)
                    d = boundary.distance(pt)
                    
                    if d < fade_distance_m:
                        # Плавний перехід (smoothstep)
                        t = d / fade_distance_m  # 0..1
                        weight = t * t * (3.0 - 2.0 * t)  # smoothstep
                        noise_weights[i] = weight
                    # else 1.0 (default - повний шум в центрі)
                    
            except Exception as e:
                print(f"[WARN] Помилка обчислення boundary masking: {e}")
                # Fallback: просто занулюємо самий край, якщо вдасться визначити
                pass

        # Генерація шуму
        np.random.seed(42)
        seed_base = int((np.sum(top_vertices_xy[:, 0]) + np.sum(top_vertices_xy[:, 1])) * 1000) % (2**31)
        np.random.seed(seed_base)
        
        noise = (np.random.random(len(top_indices)) - 0.5) * 2.0 * texture_amplitude
        
        # Застосування маски (шум тільки в центрі, краї залишаються чистими)
        masked_noise = noise * noise_weights
        
        mesh.vertices[top_indices, 2] += masked_noise
        
        # Фінальний штрих - оновити нормалі для Flat Shading
        mesh.fix_normals()

        return mesh
    except Exception as e:
        print(f"[WARN] Помилка застосування Low Poly текстури: {e}")
        import traceback
        traceback.print_exc()
        return mesh
