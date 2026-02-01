"""
Сервіс для експорту 3D моделей у формати STL та 3MF
Версія: Production Safe (Auto-naming, geometry fix)
"""
import trimesh
import trimesh.transformations
from typing import List, Optional, Tuple
import os
import numpy as np
import traceback

def _clean_mesh_geometry(mesh: trimesh.Trimesh) -> Optional[trimesh.Trimesh]:
    """Гарантує, що меш валідний, але не видаляє його при дрібних помилках"""
    if mesh is None: return None
    if isinstance(mesh, trimesh.Scene):
        # Якщо це сцена, намагаємося взяти геометрію
        if len(mesh.geometry) == 0: return None
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    
    if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0: return None
    
    try:
        # Створюємо копію
        m = mesh.copy()
        
        # Конвертація примітивів
        if hasattr(m, 'to_mesh'):
            m = m.to_mesh()
            
        # Базова валідація без агресивного process()
        if not m.is_watertight:
             # Тільки легкі виправлення
             m.merge_vertices()
             m.remove_duplicate_faces()
        
        # Перевіряємо, чи щось залишилося
        if len(m.faces) == 0:
            print("[WARN] Меш втратив усі грані після очистки. Використовуємо оригінал.")
            # Повертаємо оригінал (краще "брудний" меш, ніж нічого)
            return mesh
            
        return m
    except Exception as e:
        print(f"[WARN] Помилка обробки мешу: {e}. Використовуємо як є.")
        return mesh # Fallback до оригіналу

def export_preview_parts_stl(
    output_prefix: str,
    mesh_items: List[Tuple[str, trimesh.Trimesh]],
    model_size_mm: float = 100.0,
    add_flat_base: bool = True,
    base_thickness_mm: float = 2.0,
    rotate_to_ground: bool = False,
    reference_xy_m: Optional[Tuple[float, float]] = None,
    preserve_z: bool = False,
    preserve_xy: bool = False,
) -> dict[str, str]:
    if not mesh_items: raise ValueError("No meshes")

    working_items = []
    for n, m in mesh_items:
        clean = _clean_mesh_geometry(m)
        if clean: working_items.append((n, clean))
            
    if not working_items: raise ValueError("Empty geometry")

    combined = trimesh.util.concatenate([m for _, m in working_items])
    
    if add_flat_base:
        bounds = combined.bounds
        size = bounds[1] - bounds[0]
        center = (bounds[0] + bounds[1]) / 2.0
        base_size = [size[0], size[1], max(base_thickness_mm, 0.8)]
        min_z = bounds[0][2]
        base_z = min_z - base_size[2] / 2.0
        base_box = trimesh.creation.box(extents=base_size).to_mesh() # Force mesh
        base_box.apply_translation([center[0], center[1], base_z])
        working_items.append(("BaseFlat", base_box))
        combined = trimesh.util.concatenate([combined, base_box])

    transforms = []
    center = combined.centroid
    t0 = np.eye(4)
    if not preserve_xy:
        t0[0, 3] = -center[0]
        t0[1, 3] = -center[1]
    if not preserve_z:
        t0[2, 3] = -center[2]
    transforms.append(t0)
    combined.apply_transform(t0)

    bounds = combined.bounds
    size = bounds[1] - bounds[0]
    avg = (size[0] + size[1]) / 2.0
    if reference_xy_m:
        try: avg = (reference_xy_m[0] + reference_xy_m[1]) / 2.0
        except: pass
    
    if avg > 0:
        scale = model_size_mm / avg
        s = trimesh.transformations.scale_matrix(scale)
        transforms.append(s)
        combined.apply_transform(s)

    if rotate_to_ground:
        rot = trimesh.transformations.rotation_matrix(-np.pi/2, [1,0,0])
        transforms.append(rot)
        combined.apply_transform(rot)

    # Final center
    final_c = combined.bounds.mean(axis=0)
    final_min_z = combined.bounds[0][2]
    t_fin = np.eye(4)
    if not preserve_xy:
        t_fin[0, 3] = -final_c[0]
        t_fin[1, 3] = -final_c[1]
    if not preserve_z:
        t_fin[2, 3] = -final_min_z
    transforms.append(t_fin)

    outputs = {}
    part_map = {"Base": "base", "BaseFlat": "base", "Roads": "roads", 
                "Buildings": "buildings", "Water": "water", "Parks": "parks", "POI": "poi"}

    for name, mesh in working_items:
        key = name.split("_")[0]
        part = part_map.get(key)
        if not part: continue
        
        m_copy = mesh.copy()
        for t in transforms:
            m_copy.apply_transform(t)
            
        if part == "base" and len(m_copy.faces) > 20000:
            try: m_copy = m_copy.simplify_quadratic_decimation(15000)
            except: pass

        out = f"{output_prefix}_{part}.stl"
        m_copy.export(out)
        outputs[part] = out

    return outputs

def export_scene(
    terrain_mesh: Optional[trimesh.Trimesh],
    road_mesh: Optional[trimesh.Trimesh],
    building_meshes: List[trimesh.Trimesh],
    water_mesh: Optional[trimesh.Trimesh],
    filename: str,
    format: str = "3mf",
    model_size_mm: float = 100.0,
    add_flat_base: bool = True,
    base_thickness_mm: float = 2.0,
    parks_mesh: Optional[trimesh.Trimesh] = None,
    poi_mesh: Optional[trimesh.Trimesh] = None,
    reference_xy_m: Optional[Tuple[float, float]] = None,
    preserve_z: bool = False,
    preserve_xy: bool = False,
) -> Optional[dict]:
    
    raw_items = []
    if terrain_mesh: raw_items.append(("Base", terrain_mesh))
    if road_mesh: raw_items.append(("Roads", road_mesh))
    if water_mesh: raw_items.append(("Water", water_mesh))
    if parks_mesh: raw_items.append(("Parks", parks_mesh))
    if poi_mesh: raw_items.append(("POI", poi_mesh))
    
    if building_meshes:
        # Optimized: Combine buildings into one mesh for BOTH 3MF and STL.
        # This reduces object count from 100s to 1, fixing parsers like Bambu Studio/Win10 Viewer.
        try:
            valid_b = [b for b in building_meshes if b and len(b.vertices) > 0]
            if valid_b:
                combined_buildings = trimesh.util.concatenate(valid_b)
                raw_items.append(("Buildings", combined_buildings))
        except Exception as e:
            print(f"[WARN] Failed to combine buildings: {e}")
            # Fallback to individual if combine fails
            for i, building in enumerate(building_meshes):
                raw_items.append((f"Building_{i}", building))

    if not raw_items:
        # Fallback
        raw_items.append(("Fallback", trimesh.creation.box(extents=[10,10,1])))

    if format.lower() == "3mf":
        export_3mf(filename, raw_items, model_size_mm, add_flat_base, base_thickness_mm, False, reference_xy_m, preserve_z, preserve_xy)
    else:
        export_stl(filename, raw_items, model_size_mm, add_flat_base, base_thickness_mm, False, reference_xy_m, preserve_z, preserve_xy)

def export_3mf(
    filename: str,
    mesh_items: List[Tuple[str, trimesh.Trimesh]],
    model_size_mm: float = 100.0,
    add_flat_base: bool = True,
    base_thickness_mm: float = 2.0,
    rotate_to_ground: bool = False,
    reference_xy_m: Optional[Tuple[float, float]] = None,
    preserve_z: bool = False,
    preserve_xy: bool = False,
) -> None:
    try:
        # 1. Очистка та підготовка
        working_items = []
        for n, m in mesh_items:
            clean = _clean_mesh_geometry(m)
            if clean: working_items.append((n, clean))
            
        if not working_items: raise ValueError("No valid geometry")

        combined = trimesh.util.concatenate([m for _, m in working_items])

        if add_flat_base:
            bounds = combined.bounds
            size = bounds[1] - bounds[0]
            center = (bounds[0] + bounds[1]) / 2.0
            base_size = [size[0], size[1], max(base_thickness_mm, 0.5)]
            min_z = bounds[0][2]
            # Create BASE as MESH (important for 3mf stability)
            base_mesh = trimesh.creation.box(extents=base_size).to_mesh()
            base_mesh.apply_translation([center[0], center[1], min_z - base_size[2]/2.0])
            
            working_items.append(("BaseFlat", base_mesh))
            combined = trimesh.util.concatenate([combined, base_mesh])

        # 2. Розрахунок матриць
        transforms = []
        c = combined.centroid
        t1 = np.eye(4)
        
        # UTM logic
        abs_max = np.max(np.abs(combined.bounds))
        if abs_max > 10000 and not preserve_xy:
            t1[0,3] = -c[0]; t1[1,3] = -c[1]
        elif not preserve_xy:
            t1[0,3] = -c[0]; t1[1,3] = -c[1]
            
        if not preserve_z:
            t1[2,3] = -combined.bounds[0][2] # minZ -> 0
            
        transforms.append(t1)
        combined.apply_transform(t1)
        
        # Scale
        avg = np.mean(combined.bounds[1] - combined.bounds[0]) # simple avg
        if reference_xy_m: avg = (reference_xy_m[0] + reference_xy_m[1])/2.0
        elif (combined.bounds[1][0] - combined.bounds[0][0]) > 0:
             # Use XY avg
             size = combined.bounds[1] - combined.bounds[0]
             avg = (size[0] + size[1]) / 2.0
             
        if avg > 0:
            s = model_size_mm / avg
            t2 = trimesh.transformations.scale_matrix(s)
            transforms.append(t2)
            combined.apply_transform(t2)
            
        if rotate_to_ground:
            t3 = trimesh.transformations.rotation_matrix(-np.pi/2, [1,0,0])
            transforms.append(t3)

        # 3. Експорт Сцени (Safe Mode)
        scene = trimesh.Scene()
        
        # Color map
        colors = {
            "Base": [240, 240, 240, 255], "BaseFlat": [220, 220, 220, 255],
            "Roads": [50, 50, 50, 255], "Buildings": [180, 180, 180, 255],
            "Water": [0, 100, 255, 255], "Parks": [100, 160, 100, 255],
            "POI": [255, 200, 0, 255]
        }

        for name, mesh in working_items:
            # Apply all transforms to mesh geometry directly
            for t in transforms:
                mesh.apply_transform(t)
            
            # Apply colors safely
            key = name.split("_")[0]
            col = colors.get(key, [200,200,200,255])
            try:
                # Set face colors directly
                mesh.visual.face_colors = np.tile(col, (len(mesh.faces), 1))
            except: pass
            
            # Add to scene WITHOUT node_name to let Trimesh handle unique IDs automatically.
            # This fixes the "Duplicate ID" / "Corrupt file" error in Bambu Studio.
            scene.add_geometry(mesh)

        # Ensure directory
        d = os.path.dirname(filename)
        if d and not os.path.exists(d): os.makedirs(d, exist_ok=True)
            
        scene.export(filename)
        
    except Exception as e:
        print(f"3MF Fail: {e}")
        traceback.print_exc()
        # Fallback to STL
        export_stl(filename.replace(".3mf", ".stl"), mesh_items, model_size_mm, add_flat_base, base_thickness_mm, rotate_to_ground, reference_xy_m, preserve_z, preserve_xy)

def export_stl(
    filename: str,
    mesh_items: List[Tuple[str, trimesh.Trimesh]],
    model_size_mm: float = 100.0,
    add_flat_base: bool = True,
    base_thickness_mm: float = 2.0,
    rotate_to_ground: bool = False,
    reference_xy_m: Optional[Tuple[float, float]] = None,
    preserve_z: bool = False,
    preserve_xy: bool = False,
) -> dict:
    # 1. Clean
    valid = []
    for n, m in mesh_items:
        c = _clean_mesh_geometry(m)
        if c: valid.append((n, c))
    
    if not valid: raise ValueError("No valid geometry")
    
    # 2. Base
    combined = trimesh.util.concatenate([m for _, m in valid])
    if add_flat_base:
        b = combined.bounds
        c = (b[0]+b[1])/2
        bs = [(b[1]-b[0])[0], (b[1]-b[0])[1], max(base_thickness_mm, 0.8)]
        base = trimesh.creation.box(extents=bs).to_mesh()
        base.apply_translation([c[0], c[1], b[0][2] - bs[2]/2])
        combined = trimesh.util.concatenate([combined, base])
        valid.append(("BaseFlat", base))
        
    # 3. Transforms
    c = combined.centroid
    ox = -c[0] if not preserve_xy else 0
    oy = -c[1] if not preserve_xy else 0
    
    combined.apply_translation([ox, oy, 0])
    for _, m in valid: m.apply_translation([ox, oy, 0])
    
    # Scale
    b = combined.bounds
    sz = b[1]-b[0]
    avg = (sz[0]+sz[1])/2
    if reference_xy_m: avg = (reference_xy_m[0]+reference_xy_m[1])/2
    
    if avg > 0:
        s = model_size_mm / avg
        combined.apply_scale(s)
        for _, m in valid: m.apply_scale(s)
        
    if rotate_to_ground:
        rot = trimesh.transformations.rotation_matrix(-np.pi/2, [1,0,0])
        combined.apply_transform(rot)
        for _, m in valid: m.apply_transform(rot)
        
    if not preserve_z:
        mz = combined.bounds[0][2]
        combined.apply_translation([0,0,-mz])
        for _, m in valid: m.apply_translation([0,0,-mz])
        
    # 4. Export
    d = os.path.dirname(filename)
    if d and not os.path.exists(d): os.makedirs(d, exist_ok=True)
    combined.export(filename)
    
    # 5. Parts
    out = {}
    # Групуємо
    groups = {}
    for n, m in valid:
        k = n.split("_")[0]
        # map names
        pk = {"Base":"base","BaseFlat":"base","Roads":"roads","Buildings":"buildings",
              "Water":"water","Parks":"parks","POI":"poi"}.get(k)
        if pk:
            if pk not in groups: groups[pk] = []
            groups[pk].append(m)
            
    for k, ms in groups.items():
        try:
            p_name = filename.replace(".stl", f"_{k}.stl")
            (trimesh.util.concatenate(ms) if len(ms)>1 else ms[0]).export(p_name)
            out[k] = p_name
        except: pass
        
    return out