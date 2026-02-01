"""
Сервіс для обробки водних об'єктів з булевим відніманням
"""
import geopandas as gpd
import trimesh
import numpy as np
from shapely.geometry import Polygon, box, Point
from shapely.ops import transform
from typing import Optional
from services.terrain_provider import TerrainProvider
from services.global_center import GlobalCenter


def process_water(
    gdf_water: gpd.GeoDataFrame,
    depth_mm: float = 2.0,  # мм (для UI/сумісності)
    depth_meters: Optional[float] = None,  # якщо задано — використовуємо як "метри до масштабування"
    terrain_provider: Optional[TerrainProvider] = None,
    # backward compatibility:
    depth: Optional[float] = None,
) -> Optional[trimesh.Trimesh]:
    """
    Створює меш для води (западини для булевого віднімання)
    
    Args:
        gdf_water: GeoDataFrame з водними об'єктами
        depth: Глибина води (міліметри)
    
    Returns:
        Trimesh об'єкт води або None
    """
    if gdf_water.empty:
        return None
    
    water_meshes = []
    # ВАЖЛИВО:
    # - depth_mm у UI означає ММ НА МОДЕЛІ (після масштабування),
    # - але геометрію ми будуємо в метрах (UTM), і потім масштабуємо до мм.
    # Тому коректний шлях: main.py обчислює depth_meters і передає сюди.
    if depth is not None:
        depth_mm = float(depth)
    if depth_meters is None:
        depth_meters = depth_mm / 1000.0  # fallback (старий режим)

    # Кліп по межах рельєфу (щоб вода "не з'являлась де не треба")
    clip_box = None
    if terrain_provider is not None:
        try:
            min_x, max_x, min_y, max_y = terrain_provider.get_bounds()
            clip_box = box(min_x, min_y, max_x, max_y)
        except Exception:
            clip_box = None
    
    for idx, row in gdf_water.iterrows():
        try:
            geom = row.geometry
            
            if not geom:
                continue

            try:
                if not geom.is_valid:
                    geom = geom.buffer(0)
            except Exception:
                continue

            # Кліпимо до bbox (особливо важливо для великих water polygons, які перетинають bbox)
            if clip_box is not None:
                try:
                    geom = geom.intersection(clip_box)
                except Exception:
                    continue
                if geom.is_empty:
                    continue

            # Фільтр по площі (прибирає випадкові артефакти/дуже дрібні плями)
            try:
                if hasattr(geom, "area") and geom.area < 25.0:  # < 25 м²
                    continue
            except Exception:
                pass
            
            # Створюємо западину для води
            # Трохи спрощуємо, щоб прибрати "розпливи" від мікросегментів
            try:
                geom = geom.simplify(0.5, preserve_topology=True)
            except Exception:
                pass

            if isinstance(geom, Polygon):
                mesh = create_water_depression(geom, float(depth_meters), terrain_provider=terrain_provider)
                if mesh:
                    water_meshes.append(mesh)
            elif hasattr(geom, 'geoms'):
                for poly in geom.geoms:
                    if isinstance(poly, Polygon):
                        if hasattr(poly, "area") and poly.area < 25.0:
                            continue
                        try:
                            poly = poly.simplify(0.5, preserve_topology=True)
                        except Exception:
                            pass
                        mesh = create_water_depression(poly, float(depth_meters), terrain_provider=terrain_provider)
                        if mesh:
                            water_meshes.append(mesh)
        except Exception as e:
            print(f"Помилка обробки води {idx}: {e}")
            continue
    
    if not water_meshes:
        return None
    
    # Об'єднуємо всі водні об'єкти
    combined_water = trimesh.util.concatenate(water_meshes)
    return combined_water


def process_water_surface(
    gdf_water: gpd.GeoDataFrame,
    thickness_m: float,
    depth_meters: float,
    terrain_provider: Optional[TerrainProvider] = None,
    global_center: Optional[GlobalCenter] = None,
    surface_offset_m: float = 0.0,
) -> Optional[trimesh.Trimesh]:
    """
    Creates a thin "water surface" mesh for preview / multi-color printing.
    """
    if gdf_water is None or gdf_water.empty:
        return None
    if thickness_m <= 0:
        return None

    # Noise parameters for water texture
    ADD_TEXTURE = True
    NOISE_SCALE = 20.0  # Scale of the waves (meters)
    NOISE_AMPLITUDE = 0.15  # Height of the waves (meters) - increased for 3D printing visibility
    
    if ADD_TEXTURE:
        try:
            # Use opensimplex or simple numpy noise
            try:
                from opensimplex import OpenSimplex
                noise_gen = OpenSimplex(seed=42)
                def get_noise(x, y):
                    # 2D noise
                    return noise_gen.noise2(x / NOISE_SCALE, y / NOISE_SCALE) * NOISE_AMPLITUDE
            except ImportError:
                 # Fallback to simple sine interference
                def get_noise(x, y):
                    v1 = np.sin(x / NOISE_SCALE * 3.0)
                    v2 = np.cos(y / NOISE_SCALE * 2.5)
                    v3 = np.sin((x + y) / NOISE_SCALE * 4.0)
                    return (v1 + v2 + v3) * 0.33 * NOISE_AMPLITUDE
        except Exception:
            ADD_TEXTURE = False


    if global_center is not None:
        try:
            def to_local_transform(x, y, z=None):
                x_local, y_local = global_center.to_local(x, y)
                if z is not None:
                    return (x_local, y_local, z)
                return (x_local, y_local)
            
            gdf_water_local = gdf_water.copy()
            gdf_water_local['geometry'] = gdf_water_local['geometry'].apply(
                lambda geom: transform(to_local_transform, geom) if geom is not None and not geom.is_empty else geom
            )
            gdf_water = gdf_water_local
        except Exception as e:
            print(f"[WARN] Не вдалося перетворити gdf_water в локальні координати: {e}")

    meshes = []
    clip_box = None
    if terrain_provider is not None:
        try:
            min_x, max_x, min_y, max_y = terrain_provider.get_bounds()
            clip_box = box(min_x, min_y, max_x, max_y)
        except Exception:
            clip_box = None

    processed_count = 0
    skipped_count = 0
    
    for idx, row in gdf_water.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        
        # Validations
        if not geom.is_valid:
            geom = geom.buffer(0)
        
        if clip_box is not None:
            if not geom.intersects(clip_box):
                continue
            geom = geom.intersection(clip_box)
            if geom.is_empty:
                continue

        # Simplify to avoid excessive complexity but keep shape
        geom = geom.simplify(0.2, preserve_topology=True)

        polys = [geom] if isinstance(geom, Polygon) else list(getattr(geom, "geoms", []))
        
        for poly in polys:
            if not isinstance(poly, Polygon) or poly.is_empty:
                continue
            if poly.area < 1.0: # Skip tiny puddles
                continue
            
            try:
                # Use trimesh extrude
                mesh = trimesh.creation.extrude_polygon(poly, height=float(thickness_m))
                
                # Fix "Single Point" issue:
                # This often happens if the polygon is non-convex and triangulation is bad.
                # trimesh uses 'triangle' or 'earcut'. To improve quality, we can subdivide the mesh
                # to ensure we have internal vertices for the water texture.
                # Earcut only gives huge triangles for the top cap.
                
                # Subdivide to get more vertices for noise
                if ADD_TEXTURE:
                    # Simple subdivision might maintain bad topology. 
                    # Better: sample points inside polygon and triangulate? 
                    # Too complex. Let's precise subdivide.
                    # mesh.subdivide() works on existing faces. If top face is one huge ngon triangulated into slivers, 
                    # subdivide helps.
                    # Recalculate UVs or just use XYZ.
                    
                    # We iterate subdivision a few times to get enough density for noise
                    # Don't overdo it. 2 levels is usually enough for visual noise.
                    # Check edge lengths?
                    for _ in range(2):
                        if len(mesh.vertices) < 10000: # Limit count
                             mesh = mesh.subdivide()
                
                if mesh is None or len(mesh.vertices) == 0:
                    continue

                v = mesh.vertices.copy()
                old_z = v[:, 2].copy()
                thickness = float(thickness_m)
                
                # Identify surfaces
                is_top = old_z > (thickness * 0.9)
                is_bottom = old_z < (thickness * 0.1)
                

                if terrain_provider is not None and len(v) > 0:
                    points_array = v[:, :2]
                    
                    # 1. Get Depressed Ground (Current 'ground' after carving)
                    depressed_ground = terrain_provider.get_surface_heights_for_points(points_array)
                    
                    # 2. Get Original Ground (Shoreline reference)
                    original_ground = None
                    if hasattr(terrain_provider, 'original_heights_provider') and terrain_provider.original_heights_provider is not None:
                        # Це найнадійніший спосіб: беремо висоту до carving
                        original_ground = terrain_provider.original_heights_provider.get_surface_heights_for_points(points_array)
                    else:
                        # Fallback: припускаємо що carving був успішний
                        original_ground = depressed_ground + float(depth_meters) if depth_meters else depressed_ground + 2.0

                    # 3. Calculate Water Level per vertex
                    # Default: slightly below original ground (shoreline)
                    base_water_level = original_ground - 0.15
                    
                    # 4. Apply Noise (only to top surface)
                    height_offsets = np.zeros(len(v))
                    if ADD_TEXTURE and get_noise:
                        # Vectorized noise is hard without lib, use loop or simple numpy op
                        # Try to use numpy if possible for speed, but our get_noise is scalar
                        # Let's map it.
                        if len(v) < 50000: # Limit noise calc for performance
                             # Optimization: use numpy operations if get_noise was simple, but it is valid python func.
                             # vectorized wrapper:
                             n_v = np.vectorize(get_noise)
                             noise_val = n_v(v[:, 0], v[:, 1])
                             height_offsets = noise_val
                        else:
                             height_offsets = np.zeros(len(v))

                    # 5. Set Z Values
                    # For Top vertices:
                    # Z = max(base_water_level + noise, depressed_ground + epsilon)
                    # We strictly verify water is ABOVE the carved bed.
                    
                    top_z = base_water_level + height_offsets
                    
                    # Constraint: Water must be at least 0.2m above the bed (depressed_ground) to avoid Z-fighting.
                    # HOWEVER: If depressed_ground (current Z) is higher than original_ground (raw Z),
                    # it means a road or building was flattened ON TOP of the water (Bridge/Pier).
                    # In this case, the "bed" is effectively the top of the bridge, so we MUST NOT clamp to it.
                    
                    clamp_limit = depressed_ground + 0.2
                    
                    if original_ground is not None:
                         # Detected bridge/overpass: current ground is higher than original ground
                         # Threshold 0.5m accounts for noise/smoothing changes
                         is_bridge_or_building = (depressed_ground > (original_ground + 0.5))
                         
                         # Disable clamp for bridges (set limit extremely low)
                         clamp_limit = np.where(is_bridge_or_building, -1e9, clamp_limit)
                    
                    top_z = np.maximum(top_z, clamp_limit)
                    
                    # For Bottom vertices:
                    # just offset from top by thickness
                    # OR clamp to depressed_ground? No, we extrude down.
                    
                    # Apply
                    # We rely on is_top / is_bottom masks
                    v[is_top, 2] = top_z[is_top]
                    v[is_bottom, 2] = top_z[is_bottom] - thickness
                    
                    # Handle side vertices (interpolated between top and bottom?)
                    # Trimesh extrusion creates sides. Their Z's are old_z. 
                    # We need to stretch them.
                    # Simplest way: 
                    # new_z = bottom_z + (old_z / thickness) * (top_z - bottom_z)
                    # if old_z was 0..thickness
                    
                    # Normalize old_z [0..1]
                    # Note: old_z is 0 at bottom, 'thickness' at top usually.
                    ratio = np.clip(old_z / thickness, 0.0, 1.0)
                    
                    final_top = top_z
                    final_bottom = top_z - thickness
                    
                    v[:, 2] = final_bottom + ratio * (final_top - final_bottom)

                mesh.vertices = v
                
                # Fix color
                water_color = np.array([0, 100, 255, 255], dtype=np.uint8)
                if len(mesh.faces) > 0:
                     face_colors = np.tile(water_color, (len(mesh.faces), 1))
                     mesh.visual = trimesh.visual.ColorVisuals(face_colors=face_colors)
                
                meshes.append(mesh)
                processed_count += 1
                
            except Exception as e:
                print(f"[WARN] Failed to process water poly {idx}: {e}")
                continue

    if not meshes:
        return None
        
    try:
        combined = trimesh.util.concatenate(meshes)
        return combined
    except Exception:
        return meshes[0]


def create_water_depression(
    polygon: Polygon,
    depth: float,
    terrain_provider: Optional[TerrainProvider] = None
) -> Optional[trimesh.Trimesh]:
    """
    Створює западину для води (для булевого віднімання з бази)
    
    Args:
        polygon: Полігон води
        depth: Глибина западини (метри)
    
    Returns:
        Trimesh об'єкт западини
    """
    try:
        # Надійний шлях з підтримкою holes: trimesh.creation.extrude_polygon сам тріангулює Shapely polygon (з отворами)
        # Створює volume висотою depth над z=0 → зсуваємо вниз, щоб top був на 0.
        mesh = trimesh.creation.extrude_polygon(polygon, height=float(depth))
        mesh.apply_translation([0, 0, -float(depth)])

        # Драпіруємо на рельєф: new_z = ground_z + old_z
        if terrain_provider is not None and len(mesh.vertices) > 0:
            verts = mesh.vertices.copy()
            old_z = verts[:, 2].copy()
            ground = terrain_provider.get_surface_heights_for_points(verts[:, :2])
            verts[:, 2] = ground + old_z
            mesh.vertices = verts

        return mesh
    except Exception as e:
        print(f"Помилка створення западини води: {e}")
        return None

