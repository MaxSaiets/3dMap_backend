"""
Оптимізований solidifier рельєфу та основи.
Створює watertight mesh з основою та стінами максимальної якості.
"""
import trimesh
import numpy as np
import time
from shapely.geometry import Polygon as ShapelyPolygon
from scipy.spatial import cKDTree
from typing import Optional


def create_solid_terrain_robust(
    terrain_top: trimesh.Trimesh,
    zone_polygon: Optional[ShapelyPolygon] = None,
    base_thickness: float = 5.0,
    sampling_interval_m: float = 0.5,
    floor_z: Optional[float] = None,
    boundary_verts_3d: Optional[np.ndarray] = None,
) -> trimesh.Trimesh:
    """
    Створює solid terrain з основою та стінами максимальної якості.
    
    Args:
        terrain_top: Surface mesh (верхня поверхня рельєфу)
        zone_polygon: 2D boundary polygon (якщо None, створюється з bounds terrain_top)
        base_thickness: Товщина основи (метри)
        sampling_interval_m: Інтервал семплування вершин стін (метри)
        floor_z: Явна висота підлоги (якщо None, обчислюється з terrain)
        boundary_verts_3d: Попередньо витягнуті boundary вершини (з subdivision)
    
    Returns:
        Watertight solid mesh з основою та стінами
    """
    
    # === FIX START: Coordinate Sanity Check ===
    # Перевірка на відповідність систем координат мешу та полігона
    if zone_polygon is not None and not zone_polygon.is_empty:
        try:
            # Центр мешу (XY)
            mesh_center = terrain_top.bounds.mean(axis=0)[:2]
            # Центр полігона (XY)
            poly_bounds = zone_polygon.bounds
            poly_center = np.array([(poly_bounds[0]+poly_bounds[2])/2, (poly_bounds[1]+poly_bounds[3])/2])
            
            dist = np.linalg.norm(mesh_center - poly_center)
            
            # Якщо відстань > 5000 метрів (5 км), це явно помилка координат (Global vs Local)
            if dist > 5000:
                print(f"[CRITICAL FIX] Zone Polygon і Mesh дуже далеко один від одного (dist={dist:.1f}m)!")
                print(f"Mesh center: {mesh_center}")
                print(f"Poly center: {poly_center}")
                print("[FIX] Ігнорую zone_polygon (використовую bbox мешу), щоб уникнути розтягування.")
                zone_polygon = None # Скидаємо полігон, solidifier створить новий з bbox
        except Exception as e:
            print(f"[WARN] Coordinate sanity check failed: {e}")
    # === FIX END ===
    
    total_start = time.time()
    # print(f"[SOLIDIFIER] ПОЧАТОК create_solid_terrain_robust")
    
    # Валідація входу
    if terrain_top is None or len(terrain_top.vertices) == 0:
        raise ValueError("terrain_top порожній або невалідний")
    
    # Очищення terrain_top від NaN
    if np.any(~np.isfinite(terrain_top.vertices)):
        nan_count = np.sum(~np.isfinite(terrain_top.vertices))
        print(f"[WARN] Знайдено {nan_count} NaN/Inf в terrain_top, виправляю...")
        terrain_top.vertices = np.nan_to_num(terrain_top.vertices, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Створення zone_polygon з bounds якщо не задано
    if zone_polygon is None or zone_polygon.is_empty:
        bounds = terrain_top.bounds
        min_x, min_y = bounds[0, 0], bounds[0, 1]
        max_x, max_y = bounds[1, 0], bounds[1, 1]
        zone_polygon = ShapelyPolygon([
            (min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)
        ])
    
    # Виправлення невалідного polygon
    if not zone_polygon.is_valid:
        try:
            zone_polygon = zone_polygon.buffer(0)
            if zone_polygon.is_empty:
                # Fallback до bbox, якщо buffer(0) знищив полігон
                bounds = terrain_top.bounds
                zone_polygon = ShapelyPolygon([
                    (bounds[0,0], bounds[0,1]), (bounds[1,0], bounds[0,1]),
                    (bounds[1,0], bounds[1,1]), (bounds[0,0], bounds[1,1])
                ])
        except Exception as e:
            print(f"[WARN] Failed to fix polygon: {e}")

    # Підготовка полігона для триангуляції основи
    zone_polygon_for_base = zone_polygon
    try:
        if boundary_verts_3d is not None and len(boundary_verts_3d) >= 3:
            zone_polygon_for_base = ShapelyPolygon(np.asarray(boundary_verts_3d, dtype=float)[:, :2])
            if not zone_polygon_for_base.is_valid:
                zone_polygon_for_base = zone_polygon_for_base.buffer(0)
            if zone_polygon_for_base.is_empty:
                zone_polygon_for_base = zone_polygon
    except Exception:
        zone_polygon_for_base = zone_polygon
    
    # === 1. ОТРИМАННЯ BOUNDARY ВЕРШИН ===
    if boundary_verts_3d is not None and len(boundary_verts_3d) >= 3:
        # Використовуємо витягнуті boundary вершини (найкраща якість)
        top_wall_vertices = np.asarray(boundary_verts_3d, dtype=np.float64)
    else:
        # Семплуємо boundary з zone_polygon та проектуємо на terrain
        boundary_2d = _sample_polygon_boundary(zone_polygon, sampling_interval_m)
        boundary_z = _project_to_terrain(boundary_2d, terrain_top)
        top_wall_vertices = np.column_stack([boundary_2d, boundary_z])
    
    # Валідація top_wall_vertices
    if len(top_wall_vertices) < 3:
        raise ValueError(f"Недостатньо boundary вершин: {len(top_wall_vertices)}")
    
    if np.any(~np.isfinite(top_wall_vertices)):
        top_wall_vertices = np.nan_to_num(top_wall_vertices, nan=0.0)
    
    # === 2. ВИЗНАЧЕННЯ FLOOR Z ===
    if floor_z is None:
        min_z = float(np.min(terrain_top.vertices[:, 2]))
        floor_z = min_z - base_thickness
    else:
        floor_z = float(floor_z)
    
    # === 3. СТВОРЕННЯ СТІН ===
    n = len(top_wall_vertices)
    bottom_wall_vertices = top_wall_vertices.copy()
    bottom_wall_vertices[:, 2] = floor_z
    
    # Створення граней стін
    wall_faces = []
    for i in range(n):
        top_curr = i
        top_next = (i + 1) % n
        bot_curr = n + i
        bot_next = n + ((i + 1) % n)
        
        # T1: top_curr -> top_next -> bot_next
        wall_faces.append([top_curr, top_next, bot_next])
        # T2: top_curr -> bot_next -> bot_curr
        wall_faces.append([top_curr, bot_next, bot_curr])
    
    wall_faces = np.array(wall_faces, dtype=np.int32)
    
    # === 4. СТВОРЕННЯ ОСНОВИ ===
    base_start = time.time()
    try:
        # Спробуємо триангуляцію (Triangle engine)
        try:
            base_verts_2d, base_faces = trimesh.creation.triangulate_polygon(
                zone_polygon_for_base, engine='triangle'
            )
        except Exception:
            # Fallback: Shapely triangulation
            from shapely.ops import triangulate
            triangles = triangulate(zone_polygon_for_base)
            all_verts = []
            faces = []
            
            # Simple conversion from shapely triangles to mesh
            for tri in triangles:
                if tri.intersects(zone_polygon_for_base):
                    coords = list(tri.exterior.coords)[:-1]
                    idx_start = len(all_verts)
                    all_verts.extend(coords)
                    faces.append([idx_start, idx_start+1, idx_start+2])
            
            if len(all_verts) > 0:
                base_verts_2d = np.array(all_verts)
                base_faces = np.array(faces)
            else:
                raise ValueError("Shapely triangulation failed")

        base_verts_3d = np.column_stack([
            base_verts_2d, 
            np.full(len(base_verts_2d), floor_z, dtype=np.float64)
        ])
        
        # Перевертаємо грані щоб нормаль дивилася вниз
        base_faces = np.fliplr(base_faces)
        
    except Exception as e:
        print(f"[ERROR] Base triangulation failed: {e}. Using rectangular fallback.")
        # Fallback: простий прямокутник
        bounds = zone_polygon.bounds
        base_verts_3d = np.array([
            [bounds[0], bounds[1], floor_z],
            [bounds[2], bounds[1], floor_z],
            [bounds[2], bounds[3], floor_z],
            [bounds[0], bounds[3], floor_z]
        ])
        base_faces = np.array([[0, 2, 1], [0, 3, 2]])
    
    # === 5. З'ЄДНАННЯ ОСНОВИ ЗІ СТІНАМИ (Stitching) ===
    # Використовуємо KDTree для знаходження спільних вершин
    tree = cKDTree(bottom_wall_vertices[:, :2])
    distances, wall_indices = tree.query(base_verts_3d[:, :2], k=1)
    
    tolerance = min(sampling_interval_m * 0.3, 0.05)
    on_wall = distances < tolerance
    
    # Маппінг індексів
    num_wall = len(top_wall_vertices) + len(bottom_wall_vertices)
    interior_idx = np.where(~on_wall)[0]
    interior_verts = base_verts_3d[interior_idx]

    base_vertex_map = np.empty(len(base_verts_3d), dtype=np.int32)
    base_vertex_map[on_wall] = n + wall_indices[on_wall] # Map to bottom_wall indices
    base_vertex_map[interior_idx] = num_wall + np.arange(len(interior_idx), dtype=np.int32) # New indices

    # Перемаппінг граней основи
    base_faces_remapped = base_vertex_map[base_faces]
    
    # Створення нижньої сторони основи (Bottom Cap)
    bottom_base_vertex_map = base_vertex_map.copy()
    bottom_base_faces_remapped = bottom_base_vertex_map[np.fliplr(base_faces)]
    
    # Додаткова "fan" кришка для гарантії (інколи триангуляція пропускає краї)
    bottom_cap_faces = []
    if n >= 3:
        for i in range(1, n - 1):
            bottom_cap_faces.append([0, i, i + 1])
    bottom_cap_faces = np.asarray(bottom_cap_faces, dtype=np.int32)
    
    # === 6. ЗБИРАННЯ MESH ===
    
    # Об'єднуємо вершини стін з вершинами terrain
    terrain_vertices = terrain_top.vertices.copy()
    terrain_tree = cKDTree(terrain_vertices[:, :2])
    merge_distances, merge_indices = terrain_tree.query(top_wall_vertices[:, :2], k=1)
    merge_mask = merge_distances < tolerance
    
    wall_to_terrain_map = {}
    for i in range(len(top_wall_vertices)):
        if merge_mask[i]:
            wall_to_terrain_map[i] = merge_indices[i]
    
    # Формування єдиного масиву вершин
    all_vertices = np.vstack([
        terrain_top.vertices,
        top_wall_vertices,
        bottom_wall_vertices,
        interior_verts
    ])
    
    if np.any(~np.isfinite(all_vertices)):
        all_vertices = np.nan_to_num(all_vertices, nan=0.0)
    
    # Зсуви індексів
    terrain_vertex_count = len(terrain_top.vertices)
    
    # Ремаппінг граней стін
    if wall_to_terrain_map:
        wall_vertex_map = np.arange(n, dtype=np.int32) + terrain_vertex_count
        for wall_idx, terrain_idx in wall_to_terrain_map.items():
            wall_vertex_map[wall_idx] = terrain_idx
        
        wall_faces_remapped = wall_faces.copy()
        top_mask = wall_faces < n
        wall_faces_remapped[top_mask] = wall_vertex_map[wall_faces[top_mask]]
        wall_faces_remapped[~top_mask] = wall_faces[~top_mask] + terrain_vertex_count
        wall_faces = wall_faces_remapped
        wall_faces_offset = 0
    else:
        wall_faces_offset = terrain_vertex_count
    
    wall_faces_global = wall_faces + wall_faces_offset
    base_faces_global = base_faces_remapped # Вже використовує правильні глобальні індекси (якщо врахувати offset при маппінгу)
    # АЛЕ base_vertex_map розрахований відносно 'num_wall' (який 2*n)
    # Нам треба додати terrain_vertex_count до індексів, які посилаються на стіни та інтер'єр
    # Оскільки base_vertex_map містить індекси [n..2n] (bottom wall) та [2n..] (interior),
    # а в all_vertices вони зсунуті на terrain_vertex_count.
    
    # Виправляємо base_faces_global додаванням terrain_vertex_count
    # Оскільки wall_faces_remapped вже враховує це для стін, нам треба бути обережними.
    # Простіше перерахувати base_faces_global з нуля, знаючи структуру all_vertices:
    # 0..T-1 : Terrain
    # T..T+N-1 : Top Wall
    # T+N..T+2N-1 : Bottom Wall
    # T+2N.. : Interior
    
    # Наш base_vertex_map повертав індекси:
    # n..2n-1 -> Bottom Wall (local index)
    # 2n.. -> Interior (local index)
    
    # В all_vertices:
    # Bottom Wall починається з terrain_vertex_count + n
    # Interior починається з terrain_vertex_count + 2*n
    
    # Тому просто додаємо terrain_vertex_count до всіх індексів в base_faces_remapped
    base_faces_final = base_faces_remapped + terrain_vertex_count
    bottom_faces_final = bottom_base_faces_remapped + terrain_vertex_count
    
    if len(bottom_cap_faces) > 0:
        bottom_cap_final = bottom_cap_faces + terrain_vertex_count + n # offset to bottom wall
    else:
        bottom_cap_final = np.zeros((0, 3), dtype=np.int32)

    all_faces = np.vstack([
        terrain_top.faces,
        wall_faces_global,
        base_faces_final,
        bottom_faces_final,
        bottom_cap_final
    ])
    
    # Валідація індексів
    max_idx = len(all_vertices) - 1
    valid_mask = (all_faces >= 0) & (all_faces <= max_idx)
    all_faces = all_faces[np.all(valid_mask, axis=1)]

    # Створення mesh
    solid = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)
    
    # === 7. ОПТИМІЗАЦІЯ ТА ОЧИЩЕННЯ ===
    try:
        # Об'єднуємо вершини (Watertight glue)
        solid.merge_vertices(merge_tex=True, merge_norm=True)
        solid.remove_degenerate_faces()
        solid.remove_duplicate_faces()
        solid.remove_unreferenced_vertices()
        solid.fix_normals()
    except Exception as e:
        print(f"[WARN] Optimization failed: {e}")
    
    # Спроба зробити watertight
    if not solid.is_watertight:
        try:
            solid.fill_holes()
            if not solid.is_watertight:
                # Агресивний merge
                solid.merge_vertices(merge_tex=True, merge_norm=True)
                solid.fill_holes()
        except: pass
    
    # === FIX START: Removing Origin Artifacts ===
    # Видаляємо вершини в (0,0,0), які часто є ознакою дегенеративних трикутників або помилок нарізки
    if len(solid.vertices) > 0:
        # Шукаємо вершини дуже близькі до (0,0,0)
        origin_mask = np.all(np.abs(solid.vertices) < 1e-6, axis=1)
        if np.any(origin_mask):
            # print(f"[FIX] Removing {np.sum(origin_mask)} vertices at origin (artifacts).")
            # Видаляємо за індексами (Trimesh автоматично оновить грані)
            indices_to_remove = np.where(origin_mask)[0]
            solid.remove_indices(indices_to_remove)
    # === FIX END ===

    print(f"[SOLIDIFIER] Finished. V: {len(solid.vertices)}, F: {len(solid.faces)}, Watertight: {solid.is_watertight}")
    return solid


def _sample_polygon_boundary(polygon: ShapelyPolygon, interval: float) -> np.ndarray:
    """
    Семплує boundary polygon з рівномірними інтервалами.
    """
    if polygon is None or polygon.is_empty:
        raise ValueError("Polygon порожній або невалідний")
    
    interval = max(float(interval), 0.01)
    
    try:
        coords = list(polygon.exterior.coords)[:-1]
        if len(coords) < 3:
            raise ValueError(f"Polygon має недостатньо координат")
        
        points = np.array(coords, dtype=np.float64)
        
        # Обчислення відстаней
        segments = np.diff(points, axis=0)
        segment_lengths = np.hypot(segments[:, 0], segments[:, 1])
        cumulative = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative[-1]
        
        if total_length < 1e-6:
            return points[:, :2] # Повертаємо як є, якщо дуже малий
            
        num_samples = max(int(total_length / interval), 3)
        num_samples = min(num_samples, 20000)
        
        sample_dists = np.linspace(0, total_length, num_samples, endpoint=False)
        
        sampled_x = np.interp(sample_dists, cumulative, points[:, 0])
        sampled_y = np.interp(sample_dists, cumulative, points[:, 1])
        
        result = np.column_stack([sampled_x, sampled_y])
        return np.nan_to_num(result)
        
    except Exception as e:
        print(f"[ERROR] Sampling failed: {e}")
        bounds = polygon.bounds
        return np.array([
            [bounds[0], bounds[1]], [bounds[2], bounds[1]],
            [bounds[2], bounds[3]], [bounds[0], bounds[3]]
        ])


def _project_to_terrain(points_2d: np.ndarray, terrain: trimesh.Trimesh) -> np.ndarray:
    """
    Проектує 2D точки на terrain.
    """
    if len(points_2d) == 0: return np.array([])
    
    try:
        # Спроба використати ray casting або proximity
        pq = trimesh.proximity.ProximityQuery(terrain)
        pts3 = np.column_stack([points_2d, np.zeros(len(points_2d))])
        closest, _, _ = pq.on_surface(pts3)
        return closest[:, 2]
    except:
        # Fallback: Nearest Neighbor via KDTree
        tree = cKDTree(terrain.vertices[:, :2])
        _, idx = tree.query(points_2d, k=1)
        return terrain.vertices[idx, 2]