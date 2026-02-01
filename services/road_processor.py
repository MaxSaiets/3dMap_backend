"""
Сервіс для обробки доріг з буферизацією та об'єднанням
Покращена версія з фізичною шириною доріг та підтримкою мостів
Використовує trimesh.creation.extrude_polygon для надійної тріангуляції
"""
import osmnx as ox
import trimesh
import numpy as np
import warnings
from shapely.ops import unary_union, transform, snap
from shapely.geometry import Polygon, MultiPolygon, box, LineString, Point
from typing import Optional, List, Tuple
import geopandas as gpd
from services.terrain_provider import TerrainProvider
from services.global_center import GlobalCenter
from services.mesh_quality import improve_mesh_for_3d_printing, validate_mesh_for_3d_printing
from scipy.spatial import cKDTree

# Придушення deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pandas')


def create_bridge_supports(
    bridge_polygon: Polygon,
    bridge_height: float,
    terrain_provider: Optional[TerrainProvider],
    water_level: Optional[float],
    support_spacing: float = 20.0,  # Відстань між опорами (метри)
    support_width: float = 2.0,  # Ширина опори (метри)
    min_support_height: float = 1.0,  # Мінімальна висота опори (метри)
) -> List[trimesh.Trimesh]:
    """
    Створює опори для моста, які йдуть від моста до землі/води.
    Це необхідно для стабільності при 3D друку.
    
    Args:
        bridge_polygon: Полігон моста
        bridge_height: Висота моста (Z координата)
        terrain_provider: TerrainProvider для отримання висот землі
        water_level: Рівень води під мостом (опціонально)
        support_spacing: Відстань між опорами (метри)
        support_width: Ширина опори (метри)
        min_support_height: Мінімальна висота опори (метри)
    
    Returns:
        Список Trimesh об'єктів опор
    """
    supports = []
    
    if bridge_polygon is None or terrain_provider is None:
        return supports
    
    try:
        # Отримуємо центральну лінію моста (для розміщення опор)
        # Використовуємо centroid та bounds для визначення напрямку моста
        bounds = bridge_polygon.bounds
        minx, miny, maxx, maxy = bounds
        center_x = (minx + maxx) / 2.0
        center_y = (miny + maxy) / 2.0
        
        # Визначаємо напрямок моста (довша сторона)
        width = maxx - minx
        height = maxy - miny
        
        # ПОКРАЩЕННЯ: Розміщуємо опори по краях моста (для стабільності) + центральні для довгих мостів
        support_positions = []
        
        if width > height:
            # Міст йде вздовж X
            # Опори по краях (лівий і правий) - для стабільності
            edge_y_positions = [miny + support_width, maxy - support_width]
            
            # Центральні опори вздовж X для довгих мостів
            num_center_supports = max(0, int((width - 40) / support_spacing))  # Якщо міст довший за 40м
            if num_center_supports > 0:
                center_x_positions = np.linspace(minx + 20, maxx - 20, num_center_supports)
                # Додаємо опори на обох краях для кожної центральної позиції
                for cx in center_x_positions:
                    for ey in edge_y_positions:
                        support_positions.append((cx, ey))
            
            # Додаємо опори на початку та кінці моста (по краях)
            for ey in edge_y_positions:
                support_positions.append((minx + support_width, ey))
                support_positions.append((maxx - support_width, ey))
            
            # Якщо міст короткий - додаємо опори вздовж центральної лінії
            if width <= 40:
                num_supports = max(2, int(width / support_spacing) + 1)
                support_x_positions = np.linspace(minx + support_width, maxx - support_width, num_supports)
                for sx in support_x_positions:
                    support_positions.append((sx, center_y))
        else:
            # Міст йде вздовж Y
            # Опори по краях (верхній і нижній) - для стабільності
            edge_x_positions = [minx + support_width, maxx - support_width]
            
            # Центральні опори вздовж Y для довгих мостів
            num_center_supports = max(0, int((height - 40) / support_spacing))
            if num_center_supports > 0:
                center_y_positions = np.linspace(miny + 20, maxy - 20, num_center_supports)
                for cy in center_y_positions:
                    for ex in edge_x_positions:
                        support_positions.append((ex, cy))
            
            # Додаємо опори на початку та кінці моста (по краях)
            for ex in edge_x_positions:
                support_positions.append((ex, miny + support_width))
                support_positions.append((ex, maxy - support_width))
            
            # Якщо міст короткий - додаємо опори вздовж центральної лінії
            if height <= 40:
                num_supports = max(2, int(height / support_spacing) + 1)
                support_y_positions = np.linspace(miny + support_width, maxy - support_width, num_supports)
                for sy in support_y_positions:
                    support_positions.append((center_x, sy))
        
        # Видаляємо дублікати (якщо є)
        support_positions = list(set(support_positions))
        
        print(f"  [BRIDGE SUPPORTS] Створено {len(support_positions)} позицій опор (по краях + центральні)")
        
        # Створюємо опори
        for i, (x, y) in enumerate(support_positions):
            try:
                # Перевіряємо, чи точка всередині полігону моста
                pt = Point(x, y)
                if not bridge_polygon.contains(pt) and not bridge_polygon.touches(pt):
                    continue
                
                # ПОКРАЩЕННЯ: Семплінг висоти для площі опори (кілька точок замість однієї)
                # Це забезпечує більш точну висоту для великих опор (2м x 2м)
                support_half = support_width / 2.0
                sample_points = np.array([
                    [x - support_half, y - support_half],  # Лівий нижній кут
                    [x + support_half, y - support_half],  # Правий нижній кут
                    [x - support_half, y + support_half],  # Лівий верхній кут
                    [x + support_half, y + support_half],  # Правий верхній кут
                    [x, y]  # Центр
                ])
                
                # Отримуємо висоти для всіх точок семплінгу (по реальній поверхні terrain mesh, якщо доступно)
                ground_zs = terrain_provider.get_surface_heights_for_points(sample_points)
                ground_z = float(np.mean(ground_zs))  # Середнє значення для стабільності
                min_ground_z_sample = float(np.min(ground_zs))  # Мінімальне для перевірки води
                
                # Визначаємо висоту опори
                # Якщо є вода - опора йде до рівня води, інакше до землі
                # Використовуємо min_ground_z_sample для перевірки чи опора в воді
                if water_level is not None and min_ground_z_sample < water_level:
                    # Опора в воді - йде до рівня води
                    support_base_z = water_level
                else:
                    # Опора на землі - використовуємо середнє значення
                    support_base_z = ground_z
                
                support_height = bridge_height - support_base_z
                
                # --- ВИПРАВЛЕННЯ: Ігноруємо надто короткі опори (сміття на дорогах) ---
                # Збільшимо поріг до 4 метрів для надійності (прибирає чорні блоки на перехрестях)
                if support_height < 4.0:
                    continue
                
                # Перевіряємо мінімальну висоту
                if support_height < min_support_height:
                    # Якщо опора занадто низька, все одно створюємо її (для стабільності)
                    # Але з мінімальною висотою
                    support_height = max(min_support_height, 0.5)  # Мінімум 0.5м для видимості
                    print(f"  [BRIDGE SUPPORT] Опора {i}: висота збільшена до мінімуму {support_height:.2f}м")
                
                # Створюємо циліндричну опору
                # Використовуємо box замість cylinder для простішої геометрії (краще для 3D друку)
                support_mesh = trimesh.creation.box(
                    extents=[support_width, support_width, support_height],
                    transform=trimesh.transformations.translation_matrix([x, y, support_base_z + support_height / 2.0])
                )
                
                if support_mesh is not None and len(support_mesh.vertices) > 0:
                    # Застосовуємо сірий колір до опор (бетон/метал)
                    support_color = np.array([120, 120, 120, 255], dtype=np.uint8)  # Сірий колір
                    if len(support_mesh.faces) > 0:
                        face_colors = np.tile(support_color, (len(support_mesh.faces), 1))
                        support_mesh.visual = trimesh.visual.ColorVisuals(face_colors=face_colors)
                    supports.append(support_mesh)
                    
            except Exception as e:
                print(f"  [WARN] Помилка створення опори {i}: {e}")
                continue
        
    except Exception as e:
        print(f"  [WARN] Помилка створення опор для моста: {e}")
        import traceback
        traceback.print_exc()
    
    return supports





def detect_bridges(
    G_roads,
    water_geometries: Optional[List] = None,
    bridge_tag: str = 'bridge',
    bridge_buffer_m: float = 12.0,  # buffer around bridge centerline to mark only the bridge area
    clip_polygon: Optional[object] = None,  # Zone polygon for cross-zone bridge detection
) -> List[Tuple[object, object, float, bool, int]]:
    """
    Визначає мости: дороги, які перетинають воду або мають тег bridge=yes
    
    Args:
        G_roads: OSMnx граф доріг або GeoDataFrame
        water_geometries: Список геометрій водних об'єктів (Polygon/MultiPolygon)
        bridge_tag: Тег для визначення мостів в OSM
        
    Returns:
        List of tuples (bridge_line_geometry, bridge_area_geometry, bridge_height_offset, is_over_water, layer).
        - bridge_line_geometry: line-like geometry used for ramping (ideally LineString)
        - bridge_area_geometry: buffered polygon area used to intersect road polygons
        - bridge_height_offset: suggested lift/clearance (world meters)
        - is_over_water: True if bridge crosses water
        - layer: OSM layer value (0 = ground, 1 = first level, 2+ = higher levels)
        - start_connected_to_bridge: True if start of line connects to another bridge
        - end_connected_to_bridge: True if end of line connects to another bridge
    """
    bridges = []
    
    if G_roads is None:
        return bridges
    
    # Analyze connectivity if G_roads is a Graph
    elevated_nodes = set()
    node_bridge_count = {}
    
    is_graph = hasattr(G_roads, "nodes") and hasattr(G_roads, "edges") and not isinstance(G_roads, gpd.GeoDataFrame)
    
    if is_graph:
        try:
            # First pass: identify all bridge edges
            for u, v, k, data in G_roads.edges(keys=True, data=True):
                is_b = False
                
                # Helper to check bridge tag
                def check_tag(d):
                    val = d.get(bridge_tag)
                    if val and str(val).lower() in {"yes", "true", "1", "viaduct", "aqueduct"}:
                        return True
                    if str(d.get("bridge:structure", "")).lower() != "" or str(d.get("man_made", "")).lower() == "bridge":
                        return True
                    try:
                        if float(d.get("layer", 0)) >= 1:
                            return True
                    except:
                        pass
                    return False

                if check_tag(data):
                    node_bridge_count[u] = node_bridge_count.get(u, 0) + 1
                    node_bridge_count[v] = node_bridge_count.get(v, 0) + 1
            
            # Identify nodes where ALL incident edges (degree > 0) are bridges?
            # Or at least where there is MORE than 1 bridge connected?
            # If a node connects 2 bridges, it's an elevated joint -> no ramp.
            # If a node connects 1 bridge and 1 ground road -> it's a ramp.
            for n, count in node_bridge_count.items():
                if count >= 2:
                    elevated_nodes.add(n)
                    
        except Exception as e:
            print(f"[WARN] Connectivity analysis failed: {e}")

    # Підтримка 2 режимів
    gdf_edges = None
    if isinstance(G_roads, gpd.GeoDataFrame):
        gdf_edges = G_roads
    else:
        if not hasattr(G_roads, "edges") or len(G_roads.edges) == 0:
            return bridges
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            gdf_edges = ox.graph_to_gdfs(G_roads, nodes=False)
    
    if gdf_edges is None or gdf_edges.empty:
        return bridges
    
    # Об'єднуємо всі водні об'єкти для перевірки перетину
    water_union = None
    if water_geometries:
        try:
            water_polys = []
            for wg in water_geometries:
                if wg is not None:
                    if isinstance(wg, Polygon):
                        water_polys.append(wg)
                    elif hasattr(wg, 'geoms'):  # MultiPolygon
                        water_polys.extend(wg.geoms)
            if water_polys:
                water_union = unary_union(water_polys)
        except Exception as e:
            print(f"[WARN] Помилка об'єднання водних об'єктів для визначення мостів: {e}")
    
    # Перевіряємо кожну дорогу
    for idx, row in gdf_edges.iterrows():
        try:
            geom = row.geometry
            if geom is None:
                continue
            
            is_bridge = False
            bridge_height = 2.0  # Базова висота моста (метри)
            is_over_water = False
            layer_val = 0  # Default layer (ground level)
            
            def _is_bridge_value(v) -> bool:
                # OSMnx/GeoDataFrame can store tag values as str/bool/list.
                if v is None:
                    return False
                if isinstance(v, (list, tuple, set)):
                    return any(_is_bridge_value(x) for x in v)
                if isinstance(v, bool):
                    return bool(v)
                try:
                    s = str(v).strip().lower()
                except Exception:
                    return False
                # OSM sometimes uses "viaduct" etc as bridge values
                return s in {"yes", "true", "1", "viaduct", "aqueduct"} or s.startswith("viaduct")

            # Helper: parse numeric layer
            def _layer_value(v) -> Optional[float]:
                if v is None:
                    return None
                if isinstance(v, (list, tuple, set)):
                    for x in v:
                        lv = _layer_value(x)
                        if lv is not None:
                            return lv
                    return None
                try:
                    return float(str(v).strip())
                except Exception:
                    return None

            # 1. Перевірка тегу bridge в OSM
            if bridge_tag in row and _is_bridge_value(row.get(bridge_tag)):
                is_bridge = True
                # Визначаємо висоту моста за типом
                bridge_type = row.get('bridge:type', '')
                if 'suspension' in str(bridge_type).lower():
                    bridge_height = 5.0
                elif 'arch' in str(bridge_type).lower():
                    bridge_height = 4.0
                elif 'beam' in str(bridge_type).lower():
                    bridge_height = 3.0
                else:
                    bridge_height = 2.5

            # 1.1 Додаткові теги (інколи bridge:structure або man_made=bridge)
            if not is_bridge:
                try:
                    if _is_bridge_value(row.get("bridge:structure")) or _is_bridge_value(row.get("man_made")):
                        is_bridge = True
                        bridge_height = max(bridge_height, 2.5)
                except Exception:
                    pass

            # 1.2 Layer-based elevation (for viaducts/overpasses even without water)
            # ВИПРАВЛЕННЯ для багаторівневих розв'язок: layer=1 -> 6м, layer=2 -> 12м
            # Визначаємо layer значення для всіх доріг (навіть якщо це не міст)
            try:
                layer_raw = _layer_value(row.get("layer"))
                if layer_raw is not None:
                    layer_val = int(layer_raw)
            except Exception:
                layer_val = 0
            
            if not is_bridge:
                try:
                    layer = _layer_value(row.get("layer"))
                    if layer is not None and layer >= 1.0:
                        is_bridge = True
                        # 6м на layer для правильних шарів (layer=1 -> 6м, layer=2 -> 12м)
                        bridge_height = max(bridge_height, float(layer) * 6.0)
                except Exception:
                    pass
            
            # 1.4 CROSS-ZONE BRIDGE DETECTION
            # If road exits zone boundary towards water, classify as bridge
            # BUT only if it's a major road or has "bridge" in name
            if not is_bridge and clip_polygon is not None and water_union is not None:
                try:
                    from shapely.geometry import Point
                    # Check if road crosses zone boundary
                    if not clip_polygon.contains(geom):
                        # Road exits zone - check if it goes towards water
                        if geom.intersects(clip_polygon.boundary):
                            # Get end points
                            coords = list(geom.coords)
                            if len(coords) >= 2:
                                start_point = Point(coords[0])
                                end_point = Point(coords[-1])
                                
                                # Check if either end is outside zone and near water
                                for point in [start_point, end_point]:
                                    if not clip_polygon.contains(point):
                                        # Point outside zone - check distance to water
                                        distance_to_water = point.distance(water_union)
                                        if distance_to_water < 200.0:  # Within 200m of water
                                            # Additional check: only major roads or roads with "bridge" in name
                                            road_name = str(row.get('name', '')).lower()
                                            highway_type = str(row.get('highway', '')).lower()
                                            
                                            is_major_road = highway_type in ['motorway', 'trunk', 'primary']
                                            has_bridge_in_name = 'міст' in road_name or 'bridge' in road_name or 'патон' in road_name
                                            
                                            if is_major_road or has_bridge_in_name:
                                                is_bridge = True
                                                bridge_height = max(bridge_height, 10.0)  # Default bridge height
                                                break
                except Exception as e:
                    pass

            
            # 2. Перевірка перетину з водою
            if not is_bridge and water_union is not None:
                try:
                    # Перевіряємо чи дорога перетинає воду
                    if geom.intersects(water_union):
                        intersection_length = geom.intersection(water_union).length
                        # RELAXED: Accept 1m+ intersection (for partial bridges at zone edges)
                        if intersection_length >= 1.0:
                            is_bridge = True
                            is_over_water = True
                        else:

                            # Висота моста залежить від ширини води
                            if hasattr(water_union, 'area'):
                                # Знаходимо найближчий водний об'єкт для оцінки ширини
                                min_dist = float('inf')
                                for wg in water_geometries:
                                    if wg is not None:
                                        try:
                                            dist = geom.distance(wg)
                                            if dist < min_dist:
                                                min_dist = dist
                                                if hasattr(wg, 'bounds'):
                                                    # Оцінюємо розмір водного об'єкта
                                                    bounds = wg.bounds
                                                    width = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
                                                    if width > 50:  # Велика річка
                                                        bridge_height = 4.0
                                                    elif width > 20:  # Середня річка
                                                        bridge_height = 3.0
                                                    else:  # Мала річка
                                                        bridge_height = 2.0
                                        except:
                                            pass
                except Exception as e:
                    print(f"[WARN] Помилка перевірки перетину дороги з водою: {e}")
            
            if is_bridge:
                # IMPORTANT: return an AREA geometry for bridge marking.
                # Using raw edge LineString causes almost all buffered road polygons to "intersect a bridge".
                try:
                    bridge_line = geom
                    # If bridge is detected by water intersection, constrain to the water-crossing portion.
                    if water_union is not None:
                        try:
                            inter = geom.intersection(water_union)
                            if inter is not None and not inter.is_empty:
                                bridge_line = inter
                        except Exception:
                            pass
                    # If MultiLineString -> take the longest part for ramp projection
                    try:
                        if getattr(bridge_line, "geom_type", "") == "MultiLineString":
                            bridge_line = max(list(bridge_line.geoms), key=lambda g: getattr(g, "length", 0.0), default=geom)
                    except Exception:
                        bridge_line = geom
                    
                    # DENSIFY bridge line before buffering to allow curvature/elevation tracking
                    bridge_line = densify_geometry(bridge_line, max_segment_length=10.0)
                    
                    # Buffer into an area (polygon) so later intersection is spatially tight.
                    try:
                        bridge_area = bridge_line.buffer(float(bridge_buffer_m), cap_style=2, join_style=2, resolution=4)
                    except Exception:
                        bridge_area = bridge_line.buffer(float(bridge_buffer_m))
                    if bridge_area is not None and not bridge_area.is_empty:
                        # Корекція висоти на основі Layer
                        # Layer 1 = 6m, Layer 2 = 12m
                        final_height = max(bridge_height, float(layer_val) * 6.0)
                        if final_height < 4.0 and layer_val >= 1:
                            final_height = 4.0
                        
                        # Determine connectivity flags
                        start_elevated = False
                        end_elevated = False
                        if is_graph:
                            try:
                                # Try to match geometry endpoints to graph nodes
                                # This is tricky with simplified geometries. 
                                # We'll assume the rows came from graph_to_gdfs and index is (u, v, k)
                                if isinstance(idx, tuple) and len(idx) >= 2:
                                    u, v = idx[0], idx[1]
                                    if u in elevated_nodes:
                                        start_elevated = True
                                    if v in elevated_nodes:
                                        end_elevated = True
                            except Exception:
                                pass
                                
                        bridges.append((bridge_line, bridge_area, final_height, is_over_water, layer_val, start_elevated, end_elevated))
                except Exception:
                    # Fallback to raw geometry if buffering fails
                    try:
                        final_height = max(bridge_height, float(layer_val) * 6.0)
                        if final_height < 4.0 and layer_val >= 1:
                            final_height = 4.0
                        
                        # Densify fallback too
                        geom_d = densify_geometry(geom, max_segment_length=10.0)
                        bridges.append((geom_d, geom_d.buffer(float(bridge_buffer_m), resolution=4), final_height, is_over_water, layer_val, False, False))
                    except Exception:
                        pass
                
        except Exception as e:
            print(f"[WARN] Помилка обробки дороги для визначення моста: {e}")
            continue
    
    print(f"[INFO] Визначено {len(bridges)} мостів")
    return bridges


def densify_geometry(geom, max_segment_length=10.0):
    if geom is None or geom.is_empty:
        return geom
    
    gt = getattr(geom, "geom_type", "")
    if gt in ["LineString", "LinearRing"]:
        if geom.length <= max_segment_length:
            return geom
        
        import numpy as np
        # Calculate number of segments needed
        num_segments = int(np.ceil(geom.length / max_segment_length))
        if num_segments <= 1:
            return geom
            
        points = []
        # Interpolate points
        # For LinearRing, we want valid closed ring.
        # LineString logic works for LinearRing in shapely generally, but we must return LinearRing if input was one?
        # Actually Polygon constructor expects LinearRing or list of points.
        
        for i in range(num_segments + 1):
            fraction = float(i) / num_segments
            pt = geom.interpolate(fraction, normalized=True)
            points.append((pt.x, pt.y))
        
        from shapely.geometry import LineString, LinearRing
        if gt == "LinearRing":
            return LinearRing(points)
        return LineString(points)
        
    elif gt == "MultiLineString":
        parts = []
        for part in geom.geoms:
            parts.append(densify_geometry(part, max_segment_length))
        from shapely.geometry import MultiLineString
        return MultiLineString(parts)

    elif gt == "Polygon":
        # Densify exterior
        new_ext = densify_geometry(geom.exterior, max_segment_length)
        # Densify interiors
        new_ints = []
        for interior in geom.interiors:
            new_ints.append(densify_geometry(interior, max_segment_length))
        from shapely.geometry import Polygon
        return Polygon(new_ext, new_ints)

    elif gt == "MultiPolygon":
        parts = []
        for part in geom.geoms:
            parts.append(densify_geometry(part, max_segment_length))
        from shapely.geometry import MultiPolygon
        return MultiPolygon(parts)
        
    return geom

def build_road_polygons(
    G_roads,
    width_multiplier: float = 1.0,
    min_width_m: Optional[float] = None,
    extra_buffer_m: float = 0.0,  # Додатковий буфер з кожного боку дороги (для створення "узбіччя" при вирізанні з парків)
) -> Optional[object]:
    """
    Builds merged road polygons (2D) from a roads graph/edges gdf.
    This is useful for terrain-first operations (flattening terrain under roads) and
    also allows reusing the merged geometry for mesh generation.
    """
    if G_roads is None:
        return None

    # Support graph or edges GeoDataFrame
    gdf_edges = None
    if isinstance(G_roads, gpd.GeoDataFrame):
        gdf_edges = G_roads
    else:
        if not hasattr(G_roads, "edges") or len(G_roads.edges) == 0:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            gdf_edges = ox.graph_to_gdfs(G_roads, nodes=False)

    width_map = {
        'motorway': 12,
        'motorway_link': 10,
        'trunk': 10,
        'trunk_link': 8,
        'primary': 8,
        'primary_link': 6,
        'secondary': 6,
        'secondary_link': 5,
        'tertiary': 5,
        'tertiary_link': 4,
        'residential': 4,
        'living_street': 3,
        'service': 3,
        'unclassified': 3,
        'footway': 2,
        'path': 1.5,
        'cycleway': 2,
        'pedestrian': 2,
        'steps': 1
    }

    def get_width(row):
        highway = row.get('highway')
        if isinstance(highway, list):
            highway = highway[0] if highway else None
        elif not highway:
            return 3.0
        width = width_map.get(highway, 3.0)
        width = width * width_multiplier
        # Ensure minimum printable width (in world meters)
        try:
            if min_width_m is not None:
                width = max(float(width), float(min_width_m))
        except Exception:
            pass
        # --- ЗМІНА: Додаємо extra_buffer до ширини ---
        # Повертаємо радіус буфера (половина ширини + extra_buffer)
        return (width / 2.0) + float(extra_buffer_m)

    if 'highway' in gdf_edges.columns:
        gdf_edges = gdf_edges.copy()
        
        # DENSIFY GEOMETRY BEFORE BUFFERING
        # This allows extrusion to follow terrain curvature
        gdf_edges["geometry"] = gdf_edges["geometry"].apply(lambda g: densify_geometry(g, max_segment_length=15.0))
        
        # Calculate buffer widths
        widths = gdf_edges.apply(get_width, axis=1)
        
        # Buffer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
             # Resolution 4 is decent for road caps
            gdf_edges["geometry"] = gdf_edges.geometry.buffer(widths, cap_style=1, join_style=1, resolution=4)
    else:
        # Fallback if no highway tag
        gdf_edges = gdf_edges.copy()
        gdf_edges["geometry"] = gdf_edges["geometry"].apply(lambda g: densify_geometry(g, max_segment_length=15.0))
        width = 3.0 * width_multiplier
        if min_width_m:
            width = max(width, float(min_width_m))
        rad = (width / 2.0) + float(extra_buffer_m)
        gdf_edges["geometry"] = gdf_edges.geometry.buffer(rad, cap_style=1, join_style=1, resolution=4)

    # Merge all polygons
    try:
        merged = unary_union(gdf_edges.geometry.values)
        return merged
    except Exception as e:
        print(f"[WARN] Failed to merge road polygons: {e}")
        return None


def process_roads(
    G_roads,
    width_multiplier: float = 1.0,
    terrain_provider: Optional[TerrainProvider] = None,
    road_height: float = 1.0,  # Висота дороги у "світових" одиницях (звичайно метри в UTM-проєкції)
    road_embed: float = 0.0,   # Наскільки "втиснути" в рельєф (м), щоб гарантовано не висіла
    merged_roads: Optional[object] = None,  # Optional precomputed merged road polygons
    water_geometries: Optional[List] = None,  # Геометрії водних об'єктів для визначення мостів
    bridge_height_multiplier: float = 1.0,  # Множник для висоти мостів
    global_center: Optional[GlobalCenter] = None,  # Глобальний центр для перетворення координат
    min_width_m: Optional[float] = None,  # Мінімальна ширина дороги (в метрах у world units)
    clip_polygon: Optional[object] = None,  # Zone polygon in LOCAL coords (for pre-clipping)
    city_cache_key: Optional[str] = None,  # City cache key for cross-zone bridge detection
) -> Optional[trimesh.Trimesh]:
    """
    Обробляє дорожню мережу, створюючи 3D меші з правильною шириною
    
    Args:
        G_roads: OSMnx граф доріг
        width_multiplier: Множник для ширини доріг
    
    Returns:
        Trimesh об'єкт з об'єднаними дорогами
    """
    if G_roads is None:
        return None

    # Підтримка 2 режимів:
    # - OSMnx graph (як було)
    # - GeoDataFrame ребер (pyrosm network edges)
    gdf_edges = None
    if isinstance(G_roads, gpd.GeoDataFrame):
        gdf_edges = G_roads
    else:
        if not hasattr(G_roads, "edges") or len(G_roads.edges) == 0:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            gdf_edges = ox.graph_to_gdfs(G_roads, nodes=False)
    
    # Helper: decide if geometry looks like UTM (huge coordinates) and convert to local if needed
    def _looks_like_utm(g) -> bool:
        try:
            b = g.bounds
            return max(abs(float(b[0])), abs(float(b[1])), abs(float(b[2])), abs(float(b[3]))) > 100000.0
        except Exception:
            return False

    def _to_local_geom(g):
        if g is None or global_center is None:
            return g
        try:
            if not _looks_like_utm(g):
                return g
            def to_local_transform(x, y, z=None):
                x_local, y_local = global_center.to_local(x, y)
                if z is not None:
                    return (x_local, y_local, z)
                return (x_local, y_local)
            return transform(to_local_transform, g)
        except Exception:
            return g

    # Build or reuse merged road geometry
    if merged_roads is None:
        print("Створення буферів доріг...")
        merged_roads = build_road_polygons(G_roads, width_multiplier=width_multiplier, min_width_m=min_width_m)
    if merged_roads is None:
        return None
    
    # Ensure merged_roads are in LOCAL coords if we have global_center
    merged_roads = _to_local_geom(merged_roads)
    
    # Pre-clip to zone polygon (LOCAL coords) to prevent roads outside the zone.
    if clip_polygon is not None:
        try:
            clip_poly_local = clip_polygon
            # If clip_polygon came in UTM, convert too
            clip_poly_local = _to_local_geom(clip_poly_local)
            merged_roads = merged_roads.intersection(clip_poly_local)
            # Snap cut edges to the zone boundary to keep border coordinates stable across adjacent tiles.
            # This reduces tiny per-tile numerical differences that show up as Z seams after draping.
            try:
                merged_roads = snap(merged_roads, clip_poly_local.boundary, 1e-6)
            except Exception:
                pass
            if merged_roads is None or merged_roads.is_empty:
                return None
        except Exception:
            pass

    # Якщо є рельєф — кліпимо дороги в межі рельєфу (буферизація може виходити за bbox і давати "провали")
    if terrain_provider is not None:
        try:
            min_x, max_x, min_y, max_y = terrain_provider.get_bounds()
            # ВИПРАВЛЕННЯ: Розширюємо clip_box на 100м, щоб не втратити дороги на самому краю
            clip = box(min_x - 100.0, min_y - 100.0, max_x + 100.0, max_y + 100.0)
            merged_roads = merged_roads.intersection(clip)
            # ВИПРАВЛЕННЯ: Перевіряємо чи результат не порожній та валідний
            if merged_roads is None or merged_roads.is_empty:
                print("[WARN] Дороги стали порожніми після обрізання по рельєфу")
                return None
            # Виправляємо геометрію якщо потрібно
            if not merged_roads.is_valid:
                merged_roads = merged_roads.buffer(0)
                if merged_roads.is_empty:
                    print("[WARN] Дороги стали порожніми після виправлення")
                    return None
        except Exception as e:
            print(f"[WARN] Помилка обрізання доріг по рельєфу: {e}")
            pass

    # Конвертація в список полігонів для обробки
    if merged_roads is None or merged_roads.is_empty:
        print("[WARN] merged_roads порожній або None")
        return None
    
    if isinstance(merged_roads, Polygon):
        # Перевіряємо чи полігон має достатньо точок
        if hasattr(merged_roads, 'exterior') and len(merged_roads.exterior.coords) < 3:
            print(f"[WARN] Полігон доріг має менше 3 точок ({len(merged_roads.exterior.coords)}), пропускаємо")
            return None
        road_geoms = [merged_roads]
    elif isinstance(merged_roads, MultiPolygon):
        # Фільтруємо полігони з достатньою кількістю точок
        road_geoms = []
        for geom in merged_roads.geoms:
            if hasattr(geom, 'exterior') and len(geom.exterior.coords) >= 3:
                road_geoms.append(geom)
            else:
                print(f"[WARN] Полігон доріг має менше 3 точок, пропускаємо")
        if len(road_geoms) == 0:
            print("[WARN] Всі полігони доріг мають менше 3 точок")
            return None
    else:
        print(f"[WARN] Невідомий тип геометрії після об'єднання: {type(merged_roads)}")
        return None
    
    # ВАЖЛИВО: Перетворюємо water_geometries в локальні координати для визначення мостів
    water_geoms_local = None
    if water_geometries is not None:
        try:
            water_geoms_local = [_to_local_geom(g) for g in water_geometries if g is not None and not getattr(g, "is_empty", False)]
        except Exception:
            water_geoms_local = water_geometries

    # Ensure edges are in local coords for bridge detection (otherwise intersects never match)
    try:
        if global_center is not None and gdf_edges is not None and not gdf_edges.empty:
            # If edges look like UTM, convert to local
            sample_geom = gdf_edges.iloc[0].geometry if len(gdf_edges) else None
            if sample_geom is not None and _looks_like_utm(sample_geom):
                print(f"[DEBUG] process_roads: Converting G_roads edges from UTM to LOCAL coordinates.")
                def to_local_transform(x, y, z=None):
                    x_local, y_local = global_center.to_local(x, y)
                    if z is not None:
                        return (x_local, y_local, z)
                    return (x_local, y_local)
                gdf_edges = gdf_edges.copy()
                gdf_edges["geometry"] = gdf_edges["geometry"].apply(lambda g: transform(to_local_transform, g) if g is not None and not g.is_empty else g)
                # Also convert G_roads to edges gdf mode for bridge detection
                G_roads = gdf_edges
    except Exception:
        pass
    
    # Визначаємо мости перед обробкою (використовуємо локальні координати)
    # NOTE: detect_bridges returns bridge AREAS (buffered), not raw edge lines.
    bridges = detect_bridges(G_roads, water_geometries=water_geoms_local, clip_polygon=clip_polygon)
    # USER REQUEST: Disable all bridge processing
    bridges = []
    
    # РОЗДІЛЯЄМО МОСТИ НА КАТЕГОРІЇ
    # Layer 1 (Low): Це частина розв'язки, яка замінює дорогу на землі. Їх треба "вирізати".
    # Layer 2+ (High): Це естакади, що летять зверху. Вони НЕ повинні вирізати дороги під собою.
    bridges_low = [b for b in bridges if len(b) >= 5 and b[4] <= 1]  # Layer <= 1
    bridges_high = [b for b in bridges if len(b) >= 5 and b[4] > 1]   # Layer > 1
    
    # Маска для вирізання (тільки низькі мости + мости над водою)
    cut_mask_polys = [b[1] for b in bridges_low if b[1] is not None and not getattr(b[1], "is_empty", False)]
    # Додаємо мости над водою в маску вирізання (щоб не було доріг на дні річки)
    cut_mask_polys.extend([b[1] for b in bridges if len(b) >= 4 and b[3] and b[1] is not None and not getattr(b[1], "is_empty", False)])
    
    bridge_cut_union = None
    if cut_mask_polys:
        try:
            bridge_cut_union = unary_union(cut_mask_polys)
            if bridge_cut_union is not None and getattr(bridge_cut_union, "is_empty", False):
                bridge_cut_union = None
        except Exception:
            bridge_cut_union = None

    print(f"Створення 3D мешу доріг з {len(road_geoms)} полігонів (мостів: {len(bridges) if bridges else 0})...")
    # --- PRE-CALCULATE NODE ELEVATIONS ---
    # Create a KDTree of all graph nodes to force alignment at joints.
    node_tree = None
    node_elevations = None
    if G_roads is not None and terrain_provider is not None:
        try:
            # Extract nodes
            # If G_roads is GDF, we might not have nodes directly.
            # But we can extract endpoints of lines.
            points = []
            
            # Use raw G_roads if it is a graph
            if hasattr(G_roads, "nodes") and hasattr(G_roads, "graph"):
                 for n, d in G_roads.nodes(data=True):
                     if "x" in d and "y" in d:
                         points.append([d["x"], d["y"]])
            # Else if GDF, extract from geometry (less reliable for connectivity but helps geometry alignment)
            elif gdf_edges is not None:
                for idx, row in gdf_edges.iterrows():
                    if row.geometry:
                        if row.geometry.geom_type == 'LineString':
                            points.append(row.geometry.coords[0])
                            points.append(row.geometry.coords[-1])
                        elif row.geometry.geom_type == 'MultiLineString':
                             for g in row.geometry.geoms:
                                points.append(g.coords[0])
                                points.append(g.coords[-1])
            
            if points:
                points_arr = np.array(points)
                # Remove duplicates
                points_arr = np.unique(points_arr, axis=0)
                
                if len(points_arr) > 0:
                    node_tree = cKDTree(points_arr)
                    # Get terrain height for all nodes
                    node_zs = terrain_provider.get_surface_heights_for_points(points_arr)
                    node_elevations = node_zs # This is the RAW ground height at the node
                    print(f"[ROAD] Built Node Elevation Map for {len(points_arr)} nodes.")
        except Exception as e:
            print(f"[WARN] Failed to build node elevation map: {e}")

    road_meshes = []
    
    # Statistics
    stats = {'bridge': 0, 'ground': 0, 'anti_drown': 0}
    
    # --- DENSIFICATION STEP ---
    # Ensure all polygons are densified before extrusion to allow terrain draping
    # This recovers vertices lost during unary_union
    print(f"Densifying {len(road_geoms)} road polygons...")
    road_geoms_densified = []
    for g in road_geoms:
        try:
            densified = densify_geometry(g, max_segment_length=10.0)
            # CRITICAL: Validate geometry after densification to prevent TopologyException
            # buffer(0) fixes self-intersections and invalid geometries
            if not densified.is_valid:
                densified = densified.buffer(0)
            road_geoms_densified.append(densified)
        except Exception as e:
            print(f"[WARN] Failed to densify polygon: {e}. Using original.")
            road_geoms_densified.append(g)
    
    road_geoms = road_geoms_densified

    for poly in road_geoms:
        try:
            # Використовуємо trimesh.creation.extrude_polygon для надійної екструзії
            # Це автоматично обробляє дірки (holes) та правильно тріангулює
            try:
                def _iter_polys(g):
                    if g is None or getattr(g, "is_empty", False):
                        return []
                    gt = getattr(g, "geom_type", "")
                    if gt == "Polygon":
                        return [g]
                    if gt == "MultiPolygon":
                        return list(g.geoms)
                    if gt == "GeometryCollection":
                        return [gg for gg in g.geoms if getattr(gg, "geom_type", "") == "Polygon"]
                    return []

                def _process_one(poly_part: Polygon, is_bridge: bool, bridge_height_offset: float, bridge_line=None, start_elevated=False, end_elevated=False, layer=0):
                        # embed not > road height
                    rh = max(float(road_height), 0.0001)
                    re = float(road_embed) if road_embed is not None else 0.0
                    re = max(0.0, min(re, rh * 0.8))

                    if poly_part is None or poly_part.is_empty:
                        return

                    # Clean polygon if needed
                    try:
                        if not poly_part.is_valid:
                            poly_part = poly_part.buffer(0)
                        if poly_part.is_empty:
                            return
                        if hasattr(poly_part, "exterior") and len(poly_part.exterior.coords) < 3:
                            return
                    except Exception:
                        return

                    mesh = trimesh.creation.extrude_polygon(poly_part, height=rh)
                    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                        print(f"  [WARN] Extrude failed: mesh={'None' if mesh is None else f'{len(mesh.vertices)}v, {len(mesh.faces)}f'}, poly_area={poly_part.area:.2f}, is_bridge={is_bridge}")
                        return
                    
                    
                    print(f"  [DEBUG] Extruded: {len(mesh.vertices)}v, {len(mesh.faces)}f, is_bridge={is_bridge}, bridge_offset={bridge_height_offset:.2f}m")
                    
                    # CRITICAL: Fix normals immediately for bridges to ensure proper rendering
                    if is_bridge:
                        try:
                            mesh.fix_normals()
                            # Ensure consistent winding order
                            if not mesh.is_winding_consistent:
                                mesh.fix_normals()
                            print(f"  [DEBUG] Bridge normals fixed, is_watertight={mesh.is_watertight}, is_winding_consistent={mesh.is_winding_consistent}")
                        except Exception as e:
                            print(f"  [WARN] Failed to fix bridge normals: {e}")

                    # Project onto terrain
                    if terrain_provider is not None:
                        vertices = mesh.vertices.copy()
                        
                        # ВАЛІДАЦІЯ: Перевіряємо на NaN/Inf перед обробкою (як у H:\3dMap)
                        if np.any(~np.isfinite(vertices)):
                            print(f"  [WARN] Дорога: знайдено NaN/Inf в вершинах після екструзії")
                            vertices = np.nan_to_num(vertices, nan=0.0, posinf=1e6, neginf=-1e6)
                        
                        old_z = vertices[:, 2].copy()
                        # Get terrain height at each vertex for this polygon
                        ground_z_values = terrain_provider.get_surface_heights_for_points(vertices[:, :2])
                        
                        # ВАЛІДАЦІЯ: Перевіряємо на NaN/Inf у висотах рельєфу (як у H:\3dMap)
                        if np.any(np.isnan(ground_z_values)) or np.any(np.isinf(ground_z_values)):
                            nan_count = np.sum(np.isnan(ground_z_values))
                            inf_count = np.sum(np.isinf(ground_z_values))
                            print(f"  [WARN] Дорога: Terrain повернув NaN/Inf висот (NaN: {nan_count}, Inf: {inf_count}), використовую ground level 0.0")
                            ground_z_values = np.zeros_like(ground_z_values)
                        
                        # --- MULTI-LEVEL ROAD ELEVATION LOGIC ---
                        # Layer 0 (не bridge) → Ground road на terrain
                        # Layer 1 (bridge) → Низька розв'язка/міст (+6m)
                        # Layer 2+ (bridge) → Висока естакада (+12m, +18m, etc)
                        
                        # Get OSM layer value
                        osm_layer = int(layer) if is_bridge else 0
                        
                        if not is_bridge or osm_layer == 0:
                            # GROUND ROADS - точно на terrain (як у H:\3dMap)
                            road_z = ground_z_values + old_z - (road_embed / 1000.0)  # Embed в метрах
                            
                            # ВАЛІДАЦІЯ: Перевіряємо на NaN/Inf перед застосуванням (як у H:\3dMap)
                            if np.any(np.isnan(road_z)) or np.any(np.isinf(road_z)):
                                print(f"  [WARN] Дорога: обчислення висот дало NaN/Inf, виправляю")
                                road_z = np.nan_to_num(road_z, nan=0.0, posinf=1e6, neginf=-1e6)
                            
                            vertices[:, 2] = road_z
                            
                        else:
                            # BRIDGES - підняті залежно від layer
                            
                            # Calculate base elevation for this bridge
                            if hasattr(terrain_provider, 'original_heights_provider') and terrain_provider.original_heights_provider is not None:
                                # Є water data - розраховуємо clearance
                                try:
                                    original_ground_z = terrain_provider.original_heights_provider.get_heights_for_points(vertices[:, :2])
                                    water_depth = original_ground_z - ground_z_values
                                    max_depth = np.max(water_depth)
                                    
                                    # Визначаємо базову висоту моста
                                    if max_depth > 2.0:
                                        # Міст над водою - використати clearance
                                        water_level = np.median(original_ground_z - 0.2)
                                        min_clearance = max(8.0, float(bridge_height_offset))
                                        bridge_base_z = water_level + min_clearance
                                        print(f"  [BRIDGE Layer {osm_layer}] Over water: base={bridge_base_z:.2f}m (clearance={min_clearance:.2f}m)")
                                    else:
                                        # Естакада над землею - висота базується на layer
                                        layer_height = osm_layer * 6.0  # Layer 1=6m, 2=12m, 3=18m
                                        bridge_base_z = np.median(ground_z_values) + layer_height
                                        print(f"  [BRIDGE Layer {osm_layer}] Overpass: base={bridge_base_z:.2f}m (layer_height={layer_height:.2f}m)")
                                    
                                except Exception as e:
                                    # Fallback
                                    layer_height = max(osm_layer * 6.0, 6.0)
                                    bridge_base_z = np.median(ground_z_values) + layer_height
                                    print(f"  [BRIDGE Layer {osm_layer}] Fallback: base={bridge_base_z:.2f}m")
                            else:
                                # Немає water data - використати layer height
                                layer_height = max(osm_layer * 6.0, 6.0)
                                bridge_base_z = np.median(ground_z_values) + layer_height
                                print(f"  [BRIDGE Layer {osm_layer}] No water data: base={bridge_base_z:.2f}m")
                            
                            # Apply bridge elevation with smooth ramps
                            # Calculate ramp transitions at start/end
                            ramp_t = None
                            if bridge_line is not None and hasattr(bridge_line, 'length') and bridge_line.length > 10.0:
                                try:
                                    from shapely.geometry import Point as _Pt
                                    line_len = float(bridge_line.length)
                                    ramp_len = min(line_len * 0.3, 30.0)  # 30% of bridge or 30m max
                                    
                                    # Calculate distance along bridge for each vertex
                                    xy = vertices[:, :2]
                                    distances = np.array([bridge_line.project(_Pt(x, y)) for x, y in xy])
                                    
                                    # Sigmoid ramp: 0 at ends, 1 in middle
                                    t_start = np.clip(distances / ramp_len, 0.0, 1.0)
                                    t_end = np.clip((line_len - distances) / ramp_len, 0.0, 1.0)
                                    ramp_t = np.minimum(t_start, t_end)
                                    ramp_t = ramp_t * ramp_t * (3.0 - 2.0 * ramp_t)  # Smoothstep
                                except:
                                    pass
                            
                            if ramp_t is not None:
                                # Blend: ground at ends, bridge in middle
                                base_ground = ground_z_values
                                base_bridge = np.full_like(ground_z_values, bridge_base_z)
                                base = base_ground * (1.0 - ramp_t) + base_bridge * ramp_t
                                vertices[:, 2] = base + old_z
                            else:
                                # No ramp - full bridge height
                                vertices[:, 2] = bridge_base_z + old_z
                            
                            # Create bridge supports for 3D printing
                            if bridge_base_z - np.min(ground_z_values) > 2.0:  # Only if >2m high
                                try:
                                    supports = create_bridge_supports(
                                        poly_part,
                                        bridge_base_z,
                                        terrain_provider,
                                        support_spacing=min(30.0, poly_part.area ** 0.5),  # Adaptive spacing
                                        support_width=2.0,
                                        min_support_height=2.0
                                    )
                                    if supports:
                                        road_meshes.extend(supports)
                                        print(f"  [SUPPORTS] Created {len(supports)} bridge supports")
                                except Exception as e:
                                    print(f"  [WARN] Failed to create supports: {e}")
                            # If a road is NOT detected as a bridge but runs over deep water (depression),
                            # force it to sit on the Water Surface (Original Z) instead of the Riverbed (Ground Z).
                            # This saves "undetected" bridges from spawning underwater.
                            
                            is_drowning = False
                            if hasattr(terrain_provider, 'original_heights_provider') and terrain_provider.original_heights_provider is not None:
                                try:
                                    original_ground_z = terrain_provider.original_heights_provider.get_heights_for_points(vertices[:, :2])
                                    # Calculate depression depth at each vertex
                                    water_depth_approx = original_ground_z - ground_z_values
                                    max_depth = np.max(water_depth_approx) if len(water_depth_approx) > 0 else 0.0
                                    
                                    # CRITICAL UPGRADE: If road is even SLIGHTLY underwater (> 0.1m), promote it to a BRIDGE automatically.
                                    # This ensures "floating" roads become proper thick bridge meshes.
                                    if max_depth > 0.1:
                                        print(f"  [INFO] Anti-Drown: Road is {max_depth:.1f}m underwater. PROMOTING TO BRIDGE!")
                                        # Recursively process as bridge
                                        # Uses 5.0m height offset for reasonable bridge elevation
                                        safe_bridge_height = max(5.0, max_depth + 2.0)
                                        _process_one(poly_part, is_bridge=True, bridge_height_offset=safe_bridge_height, 
                                                    bridge_line=None, start_elevated=True, end_elevated=True)
                                        return # Stop processing as ground

                                    # Standard Anti-Drown (Pontoon mode) for shallow water or noise
                                    water_mask = water_depth_approx > 0.5
                                    if np.any(water_mask):
                                        # Lift to Water Surface + Bias
                                        road_z_water = original_ground_z + old_z + GLOBAL_ROAD_LIFT + 0.2
                                        road_z[water_mask] = road_z_water[water_mask]
                                        
                                        is_drowning = True
                                        stats['anti_drown'] += 1
                                except Exception as e:
                                    print(f"  [WARN] Anti-drown check failed: {e}")

                            vertices[:, 2] = road_z
                            if not is_bridge: stats['ground'] += 1
                        
                        # Count bridge statistics (for both water-based and fallback bridges)
                        if is_bridge:
                            stats['bridge'] += 1
                        
                        # ФІНАЛЬНА ВАЛІДАЦІЯ: Перевіряємо на NaN/Inf після обчислень (як у H:\3dMap)
                        if np.any(np.isnan(vertices)) or np.any(np.isinf(vertices)):
                            print(f"  [ERROR] Дорога: вершини містять NaN/Inf після обчислень, виправляю")
                            vertices = np.nan_to_num(vertices, nan=0.0, posinf=1e6, neginf=-1e6)
                        
                        mesh.vertices = vertices
                    else:
                        if float(re) > 0:
                            vertices = mesh.vertices.copy()
                            vertices[:, 2] = vertices[:, 2] - float(re)
                            mesh.vertices = vertices
                        if is_bridge: stats['bridge'] += 1

                    # Cleanup + color
                    try:
                        mesh.fix_normals()
                        mesh.remove_duplicate_faces()
                        mesh.remove_unreferenced_vertices()
                        if not mesh.is_volume:
                            mesh.fill_holes()
                        mesh.merge_vertices(merge_tex=True, merge_norm=True)
                        print(f"  [DEBUG] After cleanup: {len(mesh.vertices)}v, {len(mesh.faces)}f")
                        
                        # Validate coordinates
                        if len(mesh.vertices) > 0:
                            has_nan = np.any(np.isnan(mesh.vertices))
                            has_inf = np.any(np.isinf(mesh.vertices))
                            if has_nan or has_inf:
                                print(f"  [ERROR] Invalid coordinates detected! NaN={has_nan}, Inf={has_inf}")
                                print(f"  [ERROR] Vertex sample: {mesh.vertices[:3]}")
                                return  # Skip this mesh
                    except Exception:
                        pass

                    if len(mesh.faces) > 0:
                        road_color = np.array([60, 60, 60, 255], dtype=np.uint8) if is_bridge else np.array([40, 40, 40, 255], dtype=np.uint8)
                        face_colors = np.tile(road_color, (len(mesh.faces), 1))
                        mesh.visual = trimesh.visual.ColorVisuals(face_colors=face_colors)

                    # Log bounds for bridges
                    if is_bridge and len(mesh.vertices) > 0:
                        bounds = mesh.bounds
                        print(f"  [DEBUG] Bridge bounds: Z from {bounds[0][2]:.2f} to {bounds[1][2]:.2f}")
                    
                    print(f"  [DEBUG] Adding mesh: {len(mesh.vertices)}v, {len(mesh.faces)}f, is_bridge={is_bridge}")
                    road_meshes.append(mesh)

                # Build parts: bridge pieces + remainder
                # --- REFACTORED ROBUST LOGIC ---
                # Instead of separating High/Low loops and trying to match intersections back to bridges,
                # we iterate bridges directly for positive parts, and use a cut-mask for ground parts.
                
                parts_to_process: List[Tuple[Polygon, bool, float, object, bool, bool, int]] = []
                
                # 1. GENERATE BRIDGE PARTS
                # Find all bridges that intersect this road polygon
                relevant_bridges = [b for b in bridges if b[1] is not None and b[1].intersects(poly)]
                
                for b in relevant_bridges:
                    try:
                        # Intersect road poly with SPECIFIC bridge area
                        # This guarantees we know exactly which bridge parameters to use
                        b_inter = poly.intersection(b[1])
                        
                        # Handle MultiPolygon result
                        for p in _iter_polys(b_inter):
                            if p.area < 0.0001: continue  # Relaxed filter from 0.01 to 0.0001
                            
                            b_line = b[0]
                            b_height = float(b[2]) * float(bridge_height_multiplier)
                            b_layer = b[4]
                            b_start_elev = b[5] if len(b) > 5 else False
                            b_end_elev = b[6] if len(b) > 6 else False
                            
                            parts_to_process.append((p, True, b_height, b_line, b_start_elev, b_end_elev, b_layer))
                    except Exception as e:
                        print(f"[WARN] Error processing specific bridge intersection: {e}")

                # 2. GENERATE GROUND PARTS
                # Ground = Poly MINUS (Low Bridges + Bridges Over Water)
                # High bridges (over land) do NOT cut the ground (road runs underneath)
                
                # Identify mask for cutting
                cut_mask_polys = []
                for b in bridges:
                    # Layer <= 1 OR Over Water = Cut Ground
                    if b[4] <= 1 or b[3]:
                        if b[1] is not None:
                            cut_mask_polys.append(b[1])
                
                if cut_mask_polys:
                    try:
                        bridge_cut_union_local = unary_union(cut_mask_polys)
                        ground_parts = poly.difference(bridge_cut_union_local)
                    except Exception as e:
                        print(f"[WARN] Error calculating ground difference: {e}")
                        ground_parts = poly # Fallback
                else:
                    ground_parts = poly
                
                for p in _iter_polys(ground_parts):
                    if p.area < 0.01: continue
                    parts_to_process.append((p, False, 0.0, None, False, False, 0))

                # Handle case where everything was cut or empty
                if not parts_to_process and not relevant_bridges:
                     parts_to_process.append((poly, False, 0.0, None, False, False, 0))
                
                # Process parts
                for part_poly, is_bridge, bridge_height_offset, bridge_line, start_elev, end_elev, layer_val in parts_to_process:
                    _process_one(part_poly, is_bridge, bridge_height_offset, bridge_line, start_elev, end_elev, layer=layer_val)
                
            except Exception as extrude_error:
                print(f"  Помилка екструзії полігону: {extrude_error}")
                # Fallback: спробуємо створити простий меш
                continue
                
        except Exception as e:
            print(f"Помилка обробки полігону дороги: {e}")
            continue
    
    if not road_meshes:
        print("Попередження: Не вдалося створити жодного мешу доріг")
        return None
    
    print(f"Створено {len(road_meshes)} мешів доріг")
    print(f"[STATS] Generated: Bridges={stats['bridge']}, Ground={stats['ground']}, Anti-Drown Activations={stats['anti_drown']}")
    
    # Об'єднання всіх мешів доріг
    print("Об'єднання мешів доріг...")
    try:
        combined_roads = trimesh.util.concatenate(road_meshes)
        print(f"Дороги об'єднано: {len(combined_roads.vertices)} вершин, {len(combined_roads.faces)} граней")
        
        # Покращення mesh для 3D принтера
        # DISABLED: This removes small faces after scaling, which deletes bridge geometry!
        # print("Покращення якості mesh для 3D принтера (Standard Mode)...")
        # combined_roads = improve_mesh_for_3d_printing(combined_roads, aggressive=False)
        
        
        # Перевірка якості
        # DISABLED: This reports warnings about small faces that are expected after scaling
        # is_valid, mesh_warnings = validate_mesh_for_3d_printing(combined_roads)
        # if mesh_warnings:
        #     print(f"[INFO] Попередження щодо якості mesh доріг:")
        #     for w in mesh_warnings:
        #         print(f"  - {w}")
        
        
        return combined_roads
    except Exception as e:
        print(f"Помилка об'єднання доріг: {e}")
        # Повертаємо перший меш якщо не вдалося об'єднати
        if road_meshes:
            return road_meshes[0]
        return None

