"""
Оптимізований генератор рельєфу та основи.
Фокус: максимальна якість mesh, відсутність багів, правильні трикутники.
"""
import numpy as np
import trimesh
import time
import math
from typing import Tuple, Optional, List
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Polygon as ShapelyPolygon
import networkx as nx

from services.global_center import GlobalCenter, get_or_create_global_center
from services.crs_utils import bbox_latlon_to_utm
from services.heightmap import (
    get_elevation_data, 
    flatten_heightfield_under_buildings, 
    flatten_heightfield_under_polygons,
    depress_heightfield_under_polygons
)
from services.solidifier_robust import create_solid_terrain_robust as create_solid_terrain
from services.solidifier_robust import _sample_polygon_boundary

from services.terrain_provider import TerrainProvider
from services.mesh_clipper import clip_mesh_to_polygon_planes


def _snap_and_extract_boundary_from_clipped(
    mesh: trimesh.Trimesh,
    polygon: ShapelyPolygon,
    tolerance: float = 0.05,
) -> Tuple[trimesh.Trimesh, Optional[np.ndarray]]:
    """
    Вирівнює boundary обрізаного mesh по прямим лініям між вершинами полігона
    ТА одразу повертає впорядкований набір boundary вершин.
    """
    if mesh is None or len(mesh.vertices) == 0:
        return mesh, None
    if polygon is None or polygon.is_empty:
        return mesh, None

    try:
        polygon_coords = list(polygon.exterior.coords)[:-1]  # без дубліката останньої точки
        poly_verts = np.asarray(polygon_coords, dtype=np.float64)
        n_poly = len(poly_verts)
        if n_poly < 3:
            return mesh, None

        # Знаходимо boundary edges (edges, які використовуються тільки один раз)
        # Використовуємо групування trimesh для точності
        edges = mesh.edges_sorted
        unique_edges = trimesh.grouping.group_rows(edges, require_count=1)
        boundary_edges = mesh.edges[unique_edges]

        if len(boundary_edges) == 0:
            return mesh, None

        # === FIX START: Фільтрація сміттєвих ребер ===
        # Ми хочемо залишити тільки ті ребра, які є частиною найбільшого контуру.
        try:
            # Будуємо граф з ребер кордону
            G = nx.Graph()
            G.add_edges_from(boundary_edges)
            
            # Знаходимо всі компоненти (окремі замкнуті або незамкнуті лінії)
            components = list(nx.connected_components(G))
            
            if len(components) > 1:
                # print(f"[FIX] Знайдено {len(components)} окремих ліній кордону. Залишаю тільки найбільшу.")
                # Вибираємо ту групу вершин, яких найбільше
                largest_comp_nodes = max(components, key=len)
                
                # Фільтруємо boundary_edges
                valid_mask = []
                for u, v in boundary_edges:
                    # Ребро валідне, якщо обидві його вершини в найбільшому компоненті
                    valid_mask.append(u in largest_comp_nodes and v in largest_comp_nodes)
                
                boundary_edges = boundary_edges[np.array(valid_mask)]
                
                if len(boundary_edges) == 0:
                    print("[WARN] Після фільтрації не залишилось ребер!")
                    return mesh, None
        except Exception as e:
             print(f"[WARN] Boundary filtering failed: {e}")
        # === FIX END ===

        boundary_vertex_indices = np.unique(boundary_edges.flatten())
        boundary_vertices = mesh.vertices[boundary_vertex_indices]

        # Готуємо масиви для збереження, до якої сторони полігона "прив'язалась" вершина
        seg_index_for_vertex = np.full(len(boundary_vertex_indices), -1, dtype=np.int32)
        t_for_vertex = np.zeros(len(boundary_vertex_indices), dtype=np.float64)

        new_vertices = mesh.vertices.copy()
        snapped_count = 0

        for i, v_idx in enumerate(boundary_vertex_indices):
            v2d = boundary_vertices[i, :2]

            best_dist = float("inf")
            best_proj = v2d
            best_seg = -1
            best_t = 0.0

            # обходимо всі сторони полігона
            for s in range(n_poly):
                p1 = poly_verts[s]
                p2 = poly_verts[(s + 1) % n_poly]

                line_vec = p2 - p1
                line_len_sq = float(np.dot(line_vec, line_vec))
                if line_len_sq < 1e-12:
                    continue

                to_v = v2d - p1
                t = float(np.clip(np.dot(to_v, line_vec) / line_len_sq, 0.0, 1.0))
                proj = p1 + t * line_vec
                dist = float(np.linalg.norm(v2d - proj))

                if dist < best_dist:
                    best_dist = dist
                    best_proj = proj
                    best_seg = s
                    best_t = t

            if best_dist < tolerance and best_seg >= 0:
                # оновлюємо XY, Z залишаємо як був
                new_vertices[v_idx, 0] = best_proj[0]
                new_vertices[v_idx, 1] = best_proj[1]
                seg_index_for_vertex[i] = best_seg
                t_for_vertex[i] = best_t
                snapped_count += 1

        snapped_mesh = trimesh.Trimesh(
            vertices=new_vertices, faces=mesh.faces.copy(), process=False
        )

        # [FIX] Do NOT cleanup here!
        # remove_unreferenced_vertices() reindexes vertices, invalidating 'boundary_vertex_indices'
        # which causes IndexError when accessing snapped_mesh.vertices[idx] later.
        # We will cleanup at the very end if needed, or rely on the caller.
        # try:
        #     snapped_mesh.remove_duplicate_faces()
        #     snapped_mesh.remove_degenerate_faces()
        #     snapped_mesh.remove_unreferenced_vertices()
        # except Exception:
        #     pass

        # print(f"[TERRAIN] Вирівняно {snapped_count}/{len(boundary_vertex_indices)} boundary вершин")

        # Формуємо впорядкований boundary вздовж сторін полігона
        # КРИТИЧНО: Явно додаємо кутові вершини, щоб уникнути заокруглення
        final_boundary_list = []
        
        # KDTree для пошуку Z для кутів (якщо їх немає в меші)
        from scipy.spatial import cKDTree
        # Використовуємо оригінальні вершини для стабільності
        tree = cKDTree(mesh.vertices[:, :2])
        
        for s in range(n_poly):
            # 1. Додаємо вершини сегмента
            mask = seg_index_for_vertex == s
            if np.any(mask):
                seg_boundary_indices = boundary_vertex_indices[mask]
                seg_t = t_for_vertex[mask]
                order = np.argsort(seg_t)
                
                sorted_indices = seg_boundary_indices[order]
                sorted_t = seg_t[order]
                
                for idx, t_val in zip(sorted_indices, sorted_t):
                    # Фільтруємо вершини, які занадто близько до кутів
                    if t_val > 0.01 and t_val < 0.99:
                        final_boundary_list.append(snapped_mesh.vertices[idx])

            # 2. Додаємо кутову вершину
            next_s = (s + 1) % n_poly
            corner_xy = poly_verts[next_s]
            
            d, idx = tree.query(corner_xy, k=1)
            corner_z = mesh.vertices[idx, 2]
            
            final_boundary_list.append([corner_xy[0], corner_xy[1], corner_z])

        boundary_verts_3d = np.array(final_boundary_list, dtype=np.float64)

        if len(boundary_verts_3d) < 3:
            return snapped_mesh, None

        return snapped_mesh, boundary_verts_3d

    except Exception as e:
        print(f"[WARN] Помилка snap+extract boundary: {e}")
        import traceback
        traceback.print_exc()
        return mesh, None


def create_grid_faces(rows: int, cols: int) -> np.ndarray:
    """
    Створює оптимізовану сітку трикутників для регулярної grid.
    """
    if rows < 2 or cols < 2:
        return np.array([], dtype=np.int64)
    
    # Vertex indices: row i, col j -> i*cols + j
    i, j = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing='ij')
    
    # Quads to triangles
    v00 = i * cols + j                    # top-left
    v01 = i * cols + (j + 1)              # top-right
    v10 = (i + 1) * cols + j              # bottom-left
    v11 = (i + 1) * cols + (j + 1)        # bottom-right
    
    # T1: v00, v10, v01
    t1 = np.stack([v00, v10, v01], axis=-1).reshape(-1, 3)
    # T2: v01, v10, v11
    t2 = np.stack([v01, v10, v11], axis=-1).reshape(-1, 3)
    
    faces = np.vstack([t1, t2])
    return faces.astype(np.int64)


def create_terrain_mesh(
    bbox_meters: Tuple[float, float, float, float],
    z_scale: float = 1.0,
    resolution: int = 150,
    latlon_bbox: Optional[Tuple[float, float, float, float]] = None,
    source_crs: Optional[object] = None,
    terrarium_zoom: int = 15,
    elevation_ref_m: Optional[float] = None,
    baseline_offset_m: float = 0.0,
    base_thickness: float = 5.0,
    flatten_buildings: bool = True,
    building_geometries: Optional[List[BaseGeometry]] = None,
    flatten_roads: bool = False,
    road_geometries: Optional[BaseGeometry] = None,
    smoothing_sigma: float = 0.0,
    water_geometries: Optional[List[BaseGeometry]] = None,
    water_depth_m: float = 0.0,
    global_center: Optional[GlobalCenter] = None,
    bbox_is_local: bool = False,
    subdivide: bool = False,
    subdivide_levels: int = 1,
    zone_polygon: Optional[BaseGeometry] = None,
    grid_step_m: Optional[float] = None,
) -> Tuple[Optional[trimesh.Trimesh], Optional[TerrainProvider]]: # <--- ВИПРАВЛЕНО TYPE HINT
    
    total_start = time.time()
    print(f"[TERRAIN] {'='*80}")
    print(f"[TERRAIN] ПОЧАТОК create_terrain_mesh")
    
    if global_center is None:
        global_center = get_or_create_global_center(bbox_latlon=latlon_bbox)

    # 1. Grid Setup
    grid_start = time.time()
    min_x, min_y, max_x, max_y = bbox_meters
    
    # КРИТИЧНО: Розширюємо bbox на 20 метрів для кращого обрізання
    padding_m = 20.0
    expanded_min_x = min_x - padding_m
    expanded_min_y = min_y - padding_m
    expanded_max_x = max_x + padding_m
    expanded_max_y = max_y + padding_m
    
    width, height = expanded_max_x - expanded_min_x, expanded_max_y - expanded_min_y
    if width <= 0 or height <= 0:
        return None, None

    if grid_step_m is not None and grid_step_m > 0:
        step = float(grid_step_m)
        aligned_min_x = math.floor(expanded_min_x / step) * step
        aligned_max_x = math.ceil(expanded_max_x / step) * step
        aligned_min_y = math.floor(expanded_min_y / step) * step
        aligned_max_y = math.ceil(expanded_max_y / step) * step
        
        x = np.arange(aligned_min_x, aligned_max_x + step * 0.1, step)
        y = np.arange(aligned_min_y, aligned_max_y + step * 0.1, step)
        nx, ny = len(x), len(y)
        print(f"[TERRAIN] ALIGNED GRID: {nx}x{ny}, step={step:.2f}m")
    else:
        aspect = width / height
        nx, ny = int(resolution), int(resolution / aspect)
        if ny < 2: ny = 2
        print(f"[TERRAIN] LEGACY GRID: {nx}x{ny}")
        x = np.linspace(expanded_min_x, expanded_max_x, nx)
        y = np.linspace(expanded_min_y, expanded_max_y, ny)

    X, Y = np.meshgrid(x, y) 

    # 2. Elevation Data
    X_fetch, Y_fetch = X, Y
    if bbox_is_local and global_center is not None:
        cx, cy = global_center.get_center_utm()
        X_fetch = X + cx
        Y_fetch = Y + cy
        source_crs = global_center.get_utm_crs()

    elevation_start = time.time()
    Z, min_elevation = get_elevation_data(
        X_fetch, Y_fetch, latlon_bbox, z_scale, source_crs, terrarium_zoom, 
        elevation_ref_m, baseline_offset_m
    )
    
    # Cleanup Z
    Z = np.asarray(Z, dtype=np.float64)
    finite_mask = np.isfinite(Z)
    if not np.all(finite_mask):
        fill_val = np.nanmin(Z[finite_mask]) if np.any(finite_mask) else 0.0
        Z[~finite_mask] = fill_val
    
    # Clip extreme values just in case
    z_median = np.nanmedian(Z)
    Z = np.clip(Z, z_median - 3000, z_median + 5000)

    # Зберігаємо оригінальні висоти
    Z_original = Z.copy()
    
    # 3. Модифікатори рельєфу
    modifiers_start = time.time()
    
    if water_geometries and water_depth_m > 0:
        Z = depress_heightfield_under_polygons(
            X, Y, Z, water_geometries, water_depth_m, 
            min_floor=(0.0 if elevation_ref_m is not None else -base_thickness)
        )
    if flatten_buildings and building_geometries:
        Z = flatten_heightfield_under_buildings(X, Y, Z, building_geometries)
    if flatten_roads and road_geometries:
        rg_list = [road_geometries] if road_geometries else []
        if rg_list: Z = flatten_heightfield_under_polygons(X, Y, Z, rg_list)
    if smoothing_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            Z = gaussian_filter(Z, sigma=smoothing_sigma)
        except: pass
    
    # 4. Створення mesh
    mesh_start = time.time()
    faces = create_grid_faces(ny, nx)
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Fix invalid vertices
    if np.any(~np.isfinite(vertices)):
        vertices = np.nan_to_num(vertices, nan=0.0)

    # Create Initial Mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    if len(mesh.faces) > 0:
        mesh.remove_degenerate_faces()
        try:
            mesh.fix_normals()
        except: pass

    # 5. Subdivision
    if subdivide and subdivide_levels > 0:
        try:
            for _ in range(subdivide_levels):
                mesh = mesh.subdivide()
        except: pass

    # 6. Provider
    provider = TerrainProvider(X, Y, Z, original_Z=Z_original)

    # 7. Solid Creation
    solid_start = time.time()
    explicit_floor_z = -base_thickness if elevation_ref_m is not None else None
    
    if zone_polygon is None or zone_polygon.is_empty:
        zone_polygon = ShapelyPolygon([
            (min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)
        ])
    
    if not zone_polygon.is_valid:
        zone_polygon = zone_polygon.buffer(0)

    boundary_verts_3d_from_mesh = None
    
    try:
        if zone_polygon is not None and not zone_polygon.is_empty:
            clipped = clip_mesh_to_polygon_planes(mesh, zone_polygon, global_center=None)
            
            if clipped is not None and len(clipped.vertices) > 0 and len(clipped.faces) > 0:
                
                # === FIX START: Очищення мешу після кліпінгу ===
                clipped.remove_unreferenced_vertices()
                
                # Захист від вершин, що "полетіли"
                try:
                    center = clipped.bounds.mean(axis=0)
                    diag = np.linalg.norm(clipped.bounds[1] - clipped.bounds[0])
                    max_dist = max(diag * 3.0, 5000.0)
                    
                    dists = np.linalg.norm(clipped.vertices - center, axis=1)
                    valid_mask = dists < max_dist
                    
                    if not np.all(valid_mask):
                        print(f"[FIX] Removed {np.sum(~valid_mask)} exploded vertices.")
                        clipped.update_vertices(valid_mask)
                        clipped.remove_unreferenced_vertices()
                except Exception as e:
                    print(f"[WARN] Distant vertex cleaning failed: {e}")

                # [FIX] Do NOT filter components! 
                # Keeping only the largest component causes holes if the terrain is naturally split (e.g. by water/roads in clipping)
                # or if clipping produced fragmented mesh. We want ALL parts.
                # try:
                #     components = clipped.split(only_watertight=False)
                #     if isinstance(components, list) and len(components) > 1:
                #         largest = max(components, key=lambda m: len(m.faces))
                #         if len(largest.faces) > 10:
                #             clipped = largest
                #             print(f"[FIX] Kept largest component, removed {len(components)-1} fragments.")
                # except Exception as e:
                #     print(f"[WARN] Component filtering failed: {e}")
                # === FIX END ===

                mesh = clipped
                
                clipped, boundary_verts_3d_from_mesh = _snap_and_extract_boundary_from_clipped(
                    clipped, zone_polygon, tolerance=0.5
                )

        # Fallback for boundary vertices
        boundary_verts_3d_final = boundary_verts_3d_from_mesh
        if boundary_verts_3d_final is None and provider is not None:
             try:
                coords = list(zone_polygon.exterior.coords)[:-1]
                b2d = np.array(coords)
                bz = provider.get_surface_heights_for_points(b2d)
                boundary_verts_3d_final = np.column_stack([b2d, bz])
             except: pass

        solid_mesh = create_solid_terrain(
            terrain_top=mesh,
            zone_polygon=zone_polygon,
            base_thickness=base_thickness,
            sampling_interval_m=0.5,
            floor_z=explicit_floor_z,
            boundary_verts_3d=boundary_verts_3d_final
        )
        
        if solid_mesh is None: return None, None
        
        print(f"[TIMING] Total create_terrain_mesh: {time.time() - total_start:.3f} s")
        
        # === FIX: ПОВЕРТАЄМО ТІЛЬКИ 2 ЗНАЧЕННЯ ===
        return solid_mesh, provider

    except Exception as e:
        print(f"[ERROR] Terrain generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None