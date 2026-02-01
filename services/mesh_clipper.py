"""
Utilities for clipping meshes using geometric shapes.
"""
import numpy as np
import trimesh
from typing import Optional, List, Tuple
from shapely.geometry import Polygon

def clip_mesh_to_bbox(
    mesh: trimesh.Trimesh,
    bbox: Tuple[float, float, float, float],
    tolerance: float = 0.001
) -> Optional[trimesh.Trimesh]:
    """Clips mesh to bbox (minx, miny, maxx, maxy) using plane slicing for exact edges."""
    if mesh is None or len(mesh.vertices) == 0: return mesh
    minx, miny, maxx, maxy = bbox
    
    # Define 4 planes (Normals pointing INWARD)
    planes = [
        # Left wall (x=minx), normal +X
        ([1, 0, 0], [minx, 0, 0]),
        # Right wall (x=maxx), normal -X
        ([-1, 0, 0], [maxx, 0, 0]),
        # Bottom wall (y=miny), normal +Y
        ([0, 1, 0], [0, miny, 0]),
        # Top wall (y=maxy), normal -Y
        ([0, -1, 0], [0, maxy, 0])
    ]
    
    current = mesh
    try:
        for normal, origin in planes:
            # slice_mesh_plane keeps the "positive" side of the normal
            current = trimesh.intersections.slice_mesh_plane(
                mesh=current, plane_normal=normal, plane_origin=origin, cap=False
            )
            if current is None or len(current.vertices) == 0: return None
        return current
    except Exception as e:
        print(f"[WARN] Bbox slice failed: {e}")
        return mesh # Fallback to original

def clip_mesh_to_polygon(
    mesh: trimesh.Trimesh,
    polygon_coords: List[Tuple[float, float]],
    global_center=None,
    tolerance: float = 0.001
) -> Optional[trimesh.Trimesh]:
    """
    ROBUST clipper:
    Keeps faces whose centroids are inside polygon.
    Edge precision limited by mesh resolution (lossy).
    Recommended for visual clipping, flattening logic, etc.
    NOT recommended for exact CAD operations unless mesh is dense.
    """
    if mesh is None or not len(mesh.vertices): return mesh
    
    # Handle shapely Polygon input
    if hasattr(polygon_coords, "exterior"):
        # It's a shapely Polygon
        polygon_coords = list(polygon_coords.exterior.coords)
    elif hasattr(polygon_coords, "coords"):
        # It's likely a LinearRing or similar
        polygon_coords = list(polygon_coords.coords)

    if not polygon_coords or len(polygon_coords) < 3: return mesh

    # 1. Prepare local polygon coords
    local_coords = []
    if global_center:
        lons = [c[0] for c in polygon_coords]
        lats = [c[1] for c in polygon_coords]
        for lon, lat in zip(lons, lats):
             try:
                 gx, gy = global_center.to_utm(lon, lat)
                 lx, ly = global_center.to_local(gx, gy)
                 local_coords.append((lx, ly))
             except: pass
    else:
        local_coords = polygon_coords

    if len(local_coords) < 3: return mesh
    
    # Ensure CCW orientation (Shapely default)
    poly = Polygon(local_coords)
    if not poly.is_valid: poly = poly.buffer(0)
    if poly.is_empty: return mesh
    
    # Extract shell coordinates
    if hasattr(poly, 'exterior'): ring = list(poly.exterior.coords)
    else: ring = list(poly.coords)
    
    # Remove collinear/close points to avoid degenerate slices
    clean_ring = []
    for p in ring:
         if not clean_ring or np.hypot(p[0]-clean_ring[-1][0], p[1]-clean_ring[-1][1]) > 0.01:
              clean_ring.append(p)
    if len(clean_ring) > 2 and clean_ring[0] == clean_ring[-1]:
         clean_ring.pop() # Remove duplicate end
         
    if len(clean_ring) < 3: return mesh

    # Check orientation (we need CCW for Left-Inside rule)
    # Signed area method
    area = 0.0
    for i in range(len(clean_ring)):
         j = (i + 1) % len(clean_ring)
         area += clean_ring[i][0] * clean_ring[j][1]
         area -= clean_ring[j][0] * clean_ring[i][1]
    
    is_ccw = area > 0
    if not is_ccw:
         clean_ring.reverse()

    # 2. Slice against every edge
    # 2. Robust 2D Clipping (Vertex Inclusion)
    # Iterative plane slicing is brittle for complex polygons (floating point errors, open meshes).
    # Instead, we mark vertices inside the polygon and keep faces where ALL vertices are inside.
    # This prevents "slicing through" triangles (leaving jagged edges), BUT since we have high-res terrain,
    # the jagged edge matches the grid resolution, which is usually acceptable and much more robust.
    # For "perfect" edges, we would need constrained Delaunay (which terrain_generator does in stitching mode).
    # This clipper is a fallback/utility.
    
    try:
        # Check vertex inclusion
        # Use ray tracing or matplotlib path or shapely (slow but reliable)
        # For speed with mpltPath:
        from matplotlib.path import Path
        path = Path(clean_ring)
        
        # Project mesh vertices to 2D
        # mesh.vertices is (N, 3)
        # pts = mesh.vertices[:, :2] # Unused, using centroids
        
        # Check inclusion
        # radius=0 mostly, or small tolerance
        # mask = path.contains_points(pts, radius=tolerance) # (unused, we use face centroids)
        
        # Filter faces: Keep face if ALL vertices are inside? 
        # Or if ANY? "All" creates a shrunk mesh (gap at border). 
        # "Any" creates an expanded mesh (sticks out).
        # Usually "All" is safer for "clipping to inside". 
        # But we want to fill the polygon.
        # Let's use "Centroid" check for faces?
        
        # Better: Centroid check.
        # Calculate face centroids
        if len(mesh.faces) > 0:
            face_centroids = mesh.vertices[mesh.faces].mean(axis=1)[:, :2]
            face_mask = path.contains_points(face_centroids, radius=tolerance)
            
            if np.any(face_mask):
                mesh.update_faces(face_mask)
                mesh.remove_unreferenced_vertices()
                return mesh
            else:
                return None # No faces inside
        else:
            return mesh

    except ImportError:
        # Fallback to shapely (slower)
        try:
             from shapely.geometry import Point
             from shapely.prepared import prep
             
             prep_poly = prep(poly)
             
             # Face centroids
             if len(mesh.faces) > 0:
                 face_centroids = mesh.vertices[mesh.faces].mean(axis=1)[:, :2]
                 face_mask = np.array([prep_poly.contains(Point(x,y)) for x,y in face_centroids], dtype=bool)
                 
                 if np.any(face_mask):
                     mesh.update_faces(face_mask)
                     mesh.remove_unreferenced_vertices()
                     return mesh
                 else:
                     return None
             return mesh
        except Exception as e:
             print(f"[WARN] Shapely clip fallback failed: {e}")
             return mesh

    except Exception as e:
        print(f"[WARN] Robust clip failed: {e}")
        return mesh


def clip_mesh_to_polygon_planes(
    mesh: trimesh.Trimesh,
    polygon_coords: List[Tuple[float, float]],
    global_center=None,
) -> Optional[trimesh.Trimesh]:
    """
    EXACT-ish clipper for terrain: clips mesh by slicing with vertical planes along polygon edges.
    This creates new vertices along the cut and removes the "sawtooth" boundary you get from
    centroid/vertex-inclusion clipping.

    Notes:
    - Keeps the inside of the polygon (assumes CCW ring).
    - cap=False because we solidify later (walls+base).
    """
    if mesh is None or len(mesh.vertices) == 0:
        return mesh

    # Handle shapely Polygon input
    if hasattr(polygon_coords, "exterior"):
        polygon_coords = list(polygon_coords.exterior.coords)
    elif hasattr(polygon_coords, "coords"):
        polygon_coords = list(polygon_coords.coords)

    if not polygon_coords or len(polygon_coords) < 3:
        return mesh

    # Prepare local coords
    if global_center:
        local_coords = []
        lons = [c[0] for c in polygon_coords]
        lats = [c[1] for c in polygon_coords]
        for lon, lat in zip(lons, lats):
            try:
                gx, gy = global_center.to_utm(lon, lat)
                lx, ly = global_center.to_local(gx, gy)
                local_coords.append((lx, ly))
            except Exception:
                pass
    else:
        local_coords = polygon_coords

    if len(local_coords) < 3:
        return mesh

    # Ensure valid CCW ring via shapely
    poly = Polygon(local_coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return mesh

    ring = list(poly.exterior.coords)
    if len(ring) > 2 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if len(ring) < 3:
        return mesh

    # Ensure CCW orientation (signed area)
    area = 0.0
    for i in range(len(ring)):
        j = (i + 1) % len(ring)
        area += ring[i][0] * ring[j][1] - ring[j][0] * ring[i][1]
    if area < 0:
        ring = list(reversed(ring))

    current = mesh
    try:
        for i in range(len(ring)):
            x1, y1 = ring[i]
            x2, y2 = ring[(i + 1) % len(ring)]
            ex = x2 - x1
            ey = y2 - y1
            n = np.array([-ey, ex, 0.0], dtype=np.float64)  # left normal => inside for CCW
            ln = float(np.linalg.norm(n[:2]))
            if ln < 1e-9:
                continue
            n /= ln
            current = trimesh.intersections.slice_mesh_plane(
                mesh=current,
                plane_normal=n,
                plane_origin=[x1, y1, 0.0],
                cap=False,
            )
            if current is None or len(current.vertices) == 0:
                return None
        # light cleanup
        current.remove_duplicate_faces()
        current.remove_degenerate_faces()
        current.remove_unreferenced_vertices()
        return current
    except Exception as e:
        print(f"[WARN] Polygon plane-slice clip failed: {e}")
        return mesh

def clip_all_meshes_to_bbox(mesh_items, bbox, tolerance=0.001):
    out = []
    for name, m in mesh_items:
         if m:
              c = clip_mesh_to_bbox(m, bbox, tolerance)
              if c and len(c.vertices): out.append((name, c))
    return out
