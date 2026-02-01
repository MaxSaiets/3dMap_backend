"""
Сервіс для експорту 3D моделей у формати STL та 3MF
Підтримка мультиколірного друку через окремі об'єкти в 3MF
"""
import trimesh
import trimesh.transformations
from typing import List, Optional, Tuple
import os
import numpy as np

def export_preview_parts_stl(
    output_prefix: str,
    mesh_items: List[Tuple[str, trimesh.Trimesh]],
    model_size_mm: float = 100.0,
    add_flat_base: bool = True,
    base_thickness_mm: float = 2.0,
    rotate_to_ground: bool = False,
    reference_xy_m: Optional[Tuple[float, float]] = None,  # (width_m, height_m) to ensure consistent scale across tiles
    preserve_z: bool = False,  # keep global Z (avoid per-tile Z centering); still lifts minZ to 0
    preserve_xy: bool = False,  # keep global XY (do NOT center each tile); used for stitching across zones
) -> dict[str, str]:
    """
    Експортує окремі STL частини для стабільного прев'ю у браузері (з кольорами на фронтенді).
    ВАЖЛИВО: усі частини отримують однакові трансформації (center/scale/minZ), щоб ідеально збігатися.

    Повертає мапу: {"base": "..._base.stl", "roads": "..._roads.stl", ...}
    """
    if not mesh_items:
        raise ValueError("Немає мешів для preview parts")

    # Робочі копії
    working_items: List[Tuple[str, trimesh.Trimesh]] = [(n, m.copy()) for n, m in mesh_items if m is not None]
    working_items = [(n, m) for n, m in working_items if len(m.vertices) > 0 and len(m.faces) > 0]
    if not working_items:
        raise ValueError("Усі preview меші порожні")

    # Об'єднуємо для розрахунку трансформацій
    combined = trimesh.util.concatenate([m for _, m in working_items])
    if combined is None or len(combined.vertices) == 0 or len(combined.faces) == 0:
        raise ValueError("Об'єднаний preview меш порожній")

    # Додаємо плоску базу ТІЛЬКИ коли рельєф вимкнено
    if add_flat_base:
        bounds_for_base = combined.bounds
        size_for_base = bounds_for_base[1] - bounds_for_base[0]
        center_for_base = (bounds_for_base[0] + bounds_for_base[1]) / 2.0
        # ВИПРАВЛЕННЯ: Не додаємо margin по боках - база точно по розміру моделі
        margin = 0.0  # Без додаткової території по боках
        base_size = [
            size_for_base[0],  # Точний розмір без мінімуму та без margin
            size_for_base[1],  # Точний розмір без мінімуму та без margin
            max(base_thickness_mm, 0.8),
        ]
        min_z = bounds_for_base[0][2]
        base_center_z = min_z - base_size[2] / 2.0
        base_box = trimesh.creation.box(
            extents=base_size,
            transform=trimesh.transformations.translation_matrix(
                [center_for_base[0], center_for_base[1], base_center_z]
            ),
        )
        working_items.append(("BaseFlat", base_box))
        combined = trimesh.util.concatenate([combined, base_box])

    # Обчислюємо послідовність трансформацій (як у 3MF експорті)
    transforms: List[np.ndarray] = []
    combined_work = combined.copy()

    # 1) Центр по centroid
    center = combined_work.centroid
    if preserve_xy and preserve_z:
        t0 = np.eye(4)
    elif preserve_xy and not preserve_z:
        t0 = trimesh.transformations.translation_matrix([0.0, 0.0, -center[2]])
        combined_work.apply_translation([0.0, 0.0, -center[2]])
    elif preserve_z:
        t0 = trimesh.transformations.translation_matrix([-center[0], -center[1], 0.0])
        combined_work.apply_translation([-center[0], -center[1], 0.0])
    else:
        t0 = trimesh.transformations.translation_matrix(-center)
        combined_work.apply_translation(-center)
    transforms.append(t0)

    # 2) Scale XY до model_size_mm
    bounds_after = combined_work.bounds
    size_after = bounds_after[1] - bounds_after[0]
    if reference_xy_m is not None:
        try:
            avg_xy_dimension = (float(reference_xy_m[0]) + float(reference_xy_m[1])) / 2.0
        except Exception:
            avg_xy_dimension = (size_after[0] + size_after[1]) / 2.0
    else:
        avg_xy_dimension = (size_after[0] + size_after[1]) / 2.0
    if avg_xy_dimension > 0:
        scale_factor = model_size_mm / avg_xy_dimension
        s = trimesh.transformations.scale_matrix(scale_factor)
        combined_work.apply_transform(s)
        transforms.append(s)

    # 3) ВАЖЛИВО: не робимо штучного Z-scale.
    # Це спотворює висоти доріг/будівель і дає ефект "дороги занадто високі".
    # Товщину бази/рельєфу забезпечуємо на етапі генерації (terrain_base_thickness_mm / BaseFlat).

    # 4) Опційний поворот (зазвичай вимкнено)
    if rotate_to_ground:
        rot_x = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        combined_work.apply_transform(rot_x)
        transforms.append(rot_x)

    # 5) Center by bounds, minZ->0, center XY only (Z=0 лишається платформою)
    final_bounds = combined_work.bounds
    final_center_from_bounds = (final_bounds[0] + final_bounds[1]) / 2.0
    if preserve_xy and preserve_z:
        t_center = np.eye(4)
    elif preserve_xy and not preserve_z:
        t_center = trimesh.transformations.translation_matrix([0.0, 0.0, -final_center_from_bounds[2]])
        combined_work.apply_translation([0.0, 0.0, -final_center_from_bounds[2]])
    elif preserve_z:
        t_center = trimesh.transformations.translation_matrix([-final_center_from_bounds[0], -final_center_from_bounds[1], 0.0])
        combined_work.apply_translation([-final_center_from_bounds[0], -final_center_from_bounds[1], 0.0])
    else:
        t_center = trimesh.transformations.translation_matrix(-final_center_from_bounds)
        combined_work.apply_translation(-final_center_from_bounds)
    transforms.append(t_center)

    # CRITICAL (stitching): when preserve_z=True we must NOT rebase each tile by its own minZ.
    # That per-tile shift breaks height continuity across neighboring zones.
    if not preserve_z:
        final_bounds_after = combined_work.bounds
        min_z2 = final_bounds_after[0][2]
        t_minz = trimesh.transformations.translation_matrix([0, 0, -min_z2])
        combined_work.apply_translation([0, 0, -min_z2])
        transforms.append(t_minz)
    else:
        transforms.append(np.eye(4))

    if not preserve_xy:
        final_centroid_before = combined_work.centroid
        t_xy = trimesh.transformations.translation_matrix([-final_centroid_before[0], -final_centroid_before[1], 0.0])
        combined_work.apply_translation([-final_centroid_before[0], -final_centroid_before[1], 0.0])
        transforms.append(t_xy)

    # Експортуємо частини
    outputs: dict[str, str] = {}
    part_map = {
        "Base": "base",
        "BaseFlat": "base",
        "Roads": "roads",
        "Buildings": "buildings",
        "Water": "water",
        "Parks": "parks",
        "POI": "poi",
    }

    for name, mesh in working_items:
        part = part_map.get(name.split("_")[0])
        if not part:
            continue
        mesh_copy = mesh.copy()
        for mat in transforms:
            mesh_copy.apply_transform(mat)
            
        # [OPTIMIZATION] Decimate heavy 'Base' mesh for preview to prevent browser download timeouts
        if part == "base" and len(mesh_copy.faces) > 20000:
            try:
                print(f"[INFO] Optimizing Base mesh for preview ({len(mesh_copy.faces)} faces)...")
                # Target ~15k faces for preview (approx 1-2MB STL)
                mesh_copy = mesh_copy.simplify_quadratic_decimation(15000)
                print(f"[INFO] Base mesh optimized: {len(mesh_copy.faces)} faces")
            except Exception as e:
                print(f"[WARN] Failed to optimize Base mesh: {e}")

        out_path = f"{output_prefix}_{part}.stl"
        mesh_copy.export(out_path, file_type="stl")
        outputs[part] = out_path

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
    """
    Експортує 3D сцену у файл
    """
    # Готуємо список об'єктів з іменами для кольорів/окремого експорту
    mesh_items: List[Tuple[str, trimesh.Trimesh]] = []
    
    # Допоміжна функція для перевірки мешу
    def add_mesh(name: str, mesh: Optional[trimesh.Trimesh]):
        if mesh is not None and len(mesh.vertices) > 0 and len(mesh.faces) > 0:
            try:
                # Спроба виправити нормалі, але не блокуємо, якщо не вийде
                mesh.fix_normals()
            except: pass
            mesh_items.append((name, mesh))
            return True
        return False

    add_mesh("Base", terrain_mesh)
    add_mesh("Roads", road_mesh)
    add_mesh("Water", water_mesh)
    add_mesh("Parks", parks_mesh)
    add_mesh("POI", poi_mesh)

    # Обробка будівель
    if building_meshes:
        if format.lower() == "3mf":
            # Для 3MF додаємо окремо
            for i, building in enumerate(building_meshes):
                add_mesh(f"Building_{i}", building)
        else:
            # Для STL об'єднуємо
            try:
                # Filter valid buildings first
                valid_b = [b for b in building_meshes if b and len(b.vertices) > 0]
                if valid_b:
                    combined_buildings = trimesh.util.concatenate(valid_b)
                    add_mesh("Buildings", combined_buildings)
            except: pass

    # Перевірка на наявність хоча б одного мешу
    if not mesh_items:
        print("[WARN] Немає геометрії для експорту. Створюємо fallback.")
        fallback = trimesh.creation.box(extents=[100, 100, 2])
        mesh_items.append(("Fallback", fallback))

    # Експорт
    if format.lower() == "3mf":
        export_3mf(
            filename, mesh_items, model_size_mm,
            add_flat_base=add_flat_base, base_thickness_mm=base_thickness_mm,
            reference_xy_m=reference_xy_m, preserve_z=preserve_z, preserve_xy=preserve_xy
        )
        return None
    elif format.lower() == "stl":
        return export_stl(
            filename, mesh_items, model_size_mm,
            add_flat_base=add_flat_base, base_thickness_mm=base_thickness_mm,
            reference_xy_m=reference_xy_m, preserve_z=preserve_z, preserve_xy=preserve_xy
        )
    else:
        raise ValueError(f"Невідомий формат: {format}")


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
    """
    Експортує сцену у формат 3MF з підтримкою окремих об'єктів
    """
    try:
        working_items = []
        
        # 1. Підготовка та валідація мешів
        for n, m in mesh_items:
            if m is None or len(m.vertices) == 0: continue
            
            mesh_copy = m.copy()
            # Базова очистка
            try:
                mesh_copy.remove_duplicate_faces()
                mesh_copy.remove_degenerate_faces()
                mesh_copy.remove_unreferenced_vertices()
            except: pass
            
            if len(mesh_copy.faces) > 0:
                working_items.append((n, mesh_copy))

        if not working_items:
            raise ValueError("Немає валідних мешів для 3MF")

        # 2. Об'єднання для розрахунку трансформацій
        combined = trimesh.util.concatenate([m for _, m in working_items])
        
        # 3. Додавання бази (якщо треба для розрахунку bounds, але ми можемо додати її як окремий working item)
        # У user snippet база додавалася ДО working items. Тут вона додається до сцени.
        # User Logic: "combined = trimesh.util.concatenate([combined, base_box])" -> adds to bounds calculation.
        
        if add_flat_base:
            bounds = combined.bounds
            size = bounds[1] - bounds[0]
            center = (bounds[0] + bounds[1]) / 2.0
            
            base_size = [size[0], size[1], max(base_thickness_mm, 0.5)]
            base_z = bounds[0][2] - base_size[2] / 2.0
            
            base_box = trimesh.creation.box(extents=base_size)
            base_box.apply_translation([center[0], center[1], base_z])
            
            working_items.append(("BaseFlat", base_box))
            combined = trimesh.util.concatenate([combined, base_box])

        # 4. Розрахунок трансформацій (Center & Scale)
        transforms = []
        
        # Центрування
        center = combined.centroid
        t_center = np.eye(4)
        
        # Логіка центрування (XY vs Z)
        if not preserve_xy:
            t_center[0, 3] = -center[0]
            t_center[1, 3] = -center[1]
        
        # Z центрування або minZ -> 0
        if not preserve_z:
            min_z = combined.bounds[0][2]
            t_center[2, 3] = -min_z
        elif not preserve_xy: 
             # Якщо Z зберігаємо (наприклад, для тайлів), але XY центруємо по centroid
             # Тоді Z не чіпаємо, або центруємо?
             # User code: t_center[2, 3] = -center[2]  <-- This centers Z globally. Use cautiously.
             # In `preserve_z` mode (tiling), we usually want Z to stay relative to sea level OR relative to tile min.
             # If preserve_z is True, we usually imply "don't flatten to 0".
             # User snippet uses: t_center[2, 3] = -center[2]. This might be aggressive? 
             # Let's stick to user snippet but verify logic.
             t_center[2, 3] = -center[2] # Center Z

        transforms.append(t_center)
        combined.apply_transform(t_center)

        # Масштабування
        bounds = combined.bounds
        size = bounds[1] - bounds[0]
        avg_dim = (size[0] + size[1]) / 2.0
        
        if reference_xy_m: 
             avg_dim = (reference_xy_m[0] + reference_xy_m[1]) / 2.0

        if avg_dim > 0:
            scale = model_size_mm / avg_dim
            s_mat = trimesh.transformations.scale_matrix(scale)
            transforms.append(s_mat)
            combined.apply_transform(s_mat)

        if rotate_to_ground:
            rot = trimesh.transformations.rotation_matrix(-np.pi/2, [1,0,0])
            transforms.append(rot)

        # 5. Створення сцени та застосування кольорів
        scene = trimesh.Scene()
        
        color_map = {
            "Base": [240, 240, 240, 255],
            "BaseFlat": [240, 240, 240, 255],
            "Roads": [40, 40, 40, 255],
            "Buildings": [180, 180, 180, 255],
            "Water": [0, 100, 255, 255],
            "Parks": [100, 160, 100, 255],
            "POI": [255, 200, 0, 255]
        }

        # Унікальні імена для сцени
        existing_names = set()

        for name, mesh in working_items:
            # Застосування трансформацій
            for t in transforms:
                mesh.apply_transform(t)
            
            # Колір
            key = name.split("_")[0]
            color = color_map.get(key, [200, 200, 200, 255])
            
            # ВАЖЛИВО: 3MF потребує face colors або vertex colors
            try:
                if len(mesh.faces) > 0:
                     face_colors = np.tile(np.array(color, dtype=np.uint8), (len(mesh.faces), 1))
                     mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=face_colors)
            except Exception as e:
                print(f"[WARN] Failed to set color for {name}: {e}")

            # Унікальне ім'я
            node_name = name
            counter = 1
            while node_name in existing_names:
                node_name = f"{name}_{counter}"
                counter += 1
            existing_names.add(node_name)

            scene.add_geometry(mesh, node_name=node_name)

        # 6. Експорт
        scene.export(filename, file_type="3mf")
        
        # Валідація розміру
        if os.path.exists(filename) and os.path.getsize(filename) < 100:
            raise ValueError("Створений 3MF файл порожній")

    except Exception as e:
        print(f"[ERROR] 3MF Export failed: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to STL
        stl_path = filename.replace(".3mf", ".stl")
        export_stl(stl_path, mesh_items, model_size_mm, add_flat_base, base_thickness_mm, rotate_to_ground, reference_xy_m, preserve_z, preserve_xy)


def export_stl(
    filename: str,
    mesh_items: List[Tuple[str, trimesh.Trimesh]],
    model_size_mm: float = 100.0,
    add_flat_base: bool = True,
    base_thickness_mm: float = 2.0,  # тонша база
    rotate_to_ground: bool = False,  # Не крутимо, щоб не ламати орієнтацію
    reference_xy_m: Optional[Tuple[float, float]] = None,
    preserve_z: bool = False,  # keep global Z (avoid per-tile Z centering); still lifts minZ to 0
    preserve_xy: bool = False,  # keep global XY (do NOT center each tile); used for stitching across zones
) -> dict:
    """
    Експортує сцену у формат STL
    """
    try:
        if not mesh_items:
            raise ValueError("Немає мешів для експорту")
        
        # Робоча копія з нормалізацією структури
        working_items: List[Tuple[str, trimesh.Trimesh]] = []
        for n, m in mesh_items:
            if m is None or len(m.vertices) == 0 or len(m.faces) == 0:
                continue
            
            # Нормалізуємо структуру faces перед об'єднанням
            mesh_copy = m.copy()
            
            # Перевіряємо та виправляємо структуру faces
            if mesh_copy.faces.ndim == 3:
                # Якщо faces має форму (N, 3, 3), перетворюємо на (N, 3)
                if mesh_copy.faces.shape[2] == 3:
                    mesh_copy.faces = mesh_copy.faces[:, 0, :].astype(np.int64)
                else:
                    mesh_copy.faces = mesh_copy.faces.reshape(-1, 3).astype(np.int64)
            elif mesh_copy.faces.ndim == 1:
                # Якщо faces має форму (N*3,), перетворюємо на (N, 3)
                if len(mesh_copy.faces) % 3 == 0:
                    mesh_copy.faces = mesh_copy.faces.reshape(-1, 3).astype(np.int64)
                else:
                    print(f"[WARN] Неправильна форма faces для {n}: {mesh_copy.faces.shape}, пропускаємо")
                    continue
            elif mesh_copy.faces.ndim == 2:
                # Правильна форма (N, 3), але перевіряємо тип
                if mesh_copy.faces.shape[1] != 3:
                    print(f"[WARN] Неправильна форма faces для {n}: {mesh_copy.faces.shape}, пропускаємо")
                    continue
                mesh_copy.faces = mesh_copy.faces.astype(np.int64)
            else:
                print(f"[WARN] Невідома форма faces для {n}: {mesh_copy.faces.shape}, пропускаємо")
                continue
            
            # Перевіряємо індекси граней
            max_face_idx = int(np.max(mesh_copy.faces)) if len(mesh_copy.faces) > 0 else -1
            if max_face_idx >= len(mesh_copy.vertices):
                # Виправляємо індекси - видаляємо невалідні грані
                valid_faces = []
                for face in mesh_copy.faces:
                    if len(face) == 3 and all(f < len(mesh_copy.vertices) for f in face):
                        valid_faces.append(face)
                
                if len(valid_faces) == 0:
                    print(f"[WARN] Немає валідних граней для {n}, пропускаємо")
                    continue
                
                mesh_copy.faces = np.array(valid_faces, dtype=np.int64)
                mesh_copy.remove_unreferenced_vertices()
            
            working_items.append((n, mesh_copy))
            
            # Debug: log bounds
            bounds = mesh_copy.bounds
            print(f"  [DEBUG] Mesh '{n}': bounds=({bounds[0][0]:.2f}, {bounds[0][1]:.2f}, {bounds[0][2]:.2f}) to ({bounds[1][0]:.2f}, {bounds[1][1]:.2f}, {bounds[1][2]:.2f})")
        
        if not working_items:
            raise ValueError("Немає валідних мешів для експорту")
        
        # Об'єднуємо всі геометрії в один меш
        print(f"Об'єднання {len(working_items)} мешів...")
        
        
        combined = trimesh.util.concatenate([m for _, m in working_items])
        print(f"[DEBUG] After concatenate: {len(combined.vertices)}v, {len(combined.faces)}f")
        
        # Перевіряємо результат
        if combined is None or len(combined.vertices) == 0 or len(combined.faces) == 0:
            raise ValueError("Об'єднаний меш порожній")
        
        print(f"Об'єднано: {len(combined.vertices)} вершин, {len(combined.faces)} граней")
        
        # Debug: bounds after concatenate
        bounds_after_concat = combined.bounds
        print(f"  [DEBUG] Bounds after concatenate: Z from {bounds_after_concat[0][2]:.2f} to {bounds_after_concat[1][2]:.2f}")
        
        # Виправляємо нормалі перед експортом (якщо можливо)
        try:
            combined.fix_normals()
            print(f"[DEBUG] After fix_normals: bounds Z from {combined.bounds[0][2]:.2f} to {combined.bounds[1][2]:.2f}")
        except Exception as e:
            print(f"[WARN] fix_normals failed: {e}")
            pass  # Якщо не вдалося, продовжуємо
        
        # Додаємо плоску базу, якщо немає рельєфу
        if add_flat_base:
            bounds_for_base = combined.bounds
            size_for_base = bounds_for_base[1] - bounds_for_base[0]
            center_for_base = (bounds_for_base[0] + bounds_for_base[1]) / 2.0
            # ВИПРАВЛЕННЯ: Не додаємо margin по боках - база точно по розміру моделі
            margin = 0.0  # Без додаткової території по боках
            base_size = [
                size_for_base[0],  # Точний розмір без мінімуму та без margin
                size_for_base[1],  # Точний розмір без мінімуму та без margin
                max(base_thickness_mm, 0.8)
            ]
            min_z = bounds_for_base[0][2]
            base_center_z = min_z - base_size[2] / 2.0
            base_box = trimesh.creation.box(
                extents=base_size,
                transform=trimesh.transformations.translation_matrix(
                    [center_for_base[0], center_for_base[1], base_center_z]
                )
            )
            working_items.append(("BaseFlat", base_box))
            combined = trimesh.util.concatenate([combined, base_box])
            print("Додано плоску базу товщиною", base_size[2], "мм (без полів по боках)")

        # Перевіряємо розміри моделі
        bounds = combined.bounds
        size = bounds[1] - bounds[0]
        center = combined.centroid
        
        # ВИПРАВЛЕННЯ: Перевіряємо, чи координати не в UTM (дуже великі числа)
        if size[0] > 100000 or size[1] > 100000:
            print("Центрування UTM координат...")
            center_x = (bounds[0][0] + bounds[1][0]) / 2.0
            center_y = (bounds[0][1] + bounds[1][1]) / 2.0
            
            vertices = combined.vertices.copy()
            vertices[:, 0] -= center_x
            vertices[:, 1] -= center_y
            # CRITICAL: process=False to avoid vertex merging with UTM tolerances
            combined = trimesh.Trimesh(vertices=vertices, faces=combined.faces, process=False)
            print(f"[DEBUG] After UTM centering: bounds Z from {combined.bounds[0][2]:.2f} to {combined.bounds[1][2]:.2f}")
            
            bounds_after_xy = combined.bounds
            min_z = bounds_after_xy[0][2]
            if not preserve_z:
                vertices = combined.vertices.copy()
                vertices[:, 2] -= min_z
                combined = trimesh.Trimesh(vertices=vertices, faces=combined.faces, process=False)
            
            bounds_after = combined.bounds
            size_after = bounds_after[1] - bounds_after[0]
            max_dimension = max(size_after[0], size_after[1])
            
            if max_dimension > 100000:
                print("Повторне центрування...")
                bounds_check = combined.bounds
                center_x = (bounds_check[0][0] + bounds_check[1][0]) / 2.0
                center_y = (bounds_check[0][1] + bounds_check[1][1]) / 2.0
                
                vertices = combined.vertices.copy()
                vertices[:, 0] -= center_x
                vertices[:, 1] -= center_y
                combined = trimesh.Trimesh(vertices=vertices, faces=combined.faces, process=False)
                
                bounds_after_xy = combined.bounds
                min_z = bounds_after_xy[0][2]
                if not preserve_z:
                    vertices = combined.vertices.copy()
                    vertices[:, 2] -= min_z
                    combined = trimesh.Trimesh(vertices=vertices, faces=combined.faces, process=False)
                
                bounds_after = combined.bounds
                size_after = bounds_after[1] - bounds_after[0]
        else:
            # Standard: center by centroid for a nice single-tile export.
            # For stitching mode we preserve global XY/Z and only do minZ->0 later.
            if preserve_xy and preserve_z:
                pass
            elif preserve_xy and not preserve_z:
                combined.apply_translation([0.0, 0.0, -center[2]])
            elif preserve_z:
                combined.apply_translation([-center[0], -center[1], 0.0])
                bounds_after = combined.bounds
                print(f"  [DEBUG] After XY centering (preserve_z): Z from {bounds_after[0][2]:.2f} to {bounds_after[1][2]:.2f}, vertices={len(combined.vertices)}")
            else:
                combined.apply_translation(-center)
                print(f"  [DEBUG] After full centering: Z from {combined.bounds[0][2]:.2f} to {combined.bounds[1][2]:.2f}")
            bounds_after = combined.bounds
            size_after = bounds_after[1] - bounds_after[0]
        
        # Масштабуємо модель до заданого розміру (в міліметрах)
        # Використовуємо середнє арифметичне X та Y для більш збалансованого масштабування
        # Це запобігає створенню дуже вузьких моделей
        if reference_xy_m is not None:
            try:
                avg_xy_dimension = (float(reference_xy_m[0]) + float(reference_xy_m[1])) / 2.0
            except Exception:
                avg_xy_dimension = (size_after[0] + size_after[1]) / 2
        else:
            avg_xy_dimension = (size_after[0] + size_after[1]) / 2
        
        # Перевіряємо, чи модель не занадто мала (менше 0.1 мм) - це може бути помилка
        if avg_xy_dimension < 0.001:
            print(f"⚠️ Попередження: Модель дуже мала ({avg_xy_dimension:.6f} мм), можливо проблема з даними")
            # Якщо модель занадто мала, не масштабуємо - залишаємо як є
        elif avg_xy_dimension > 0:
            # Конвертуємо міліметри в одиниці моделі (1 одиниця = 1 мм)
            # Масштабуємо так, щоб середній розмір X/Y був model_size_mm
            scale_factor = model_size_mm / avg_xy_dimension
            
            # Перевіряємо, чи коефіцієнт не занадто великий або малий
            if scale_factor > 1000000 or scale_factor < 0.000001:
                print(f"⚠️ Помилка масштабування: коефіцієнт {scale_factor:.6f}, розміри: {size_after}")
                scale_factor = 1.0
            else:
                # Масштабуємо ОДНАКОВО по X/Y/Z.
                # Вся геометрія в нас в "світових" одиницях (метри в локальних координатах),
                # а print-aware товщини (roads/building foundation/parks/etc) вже конвертовані в метри через /scale_factor.
                # Тому anisotropic XY-only scale робить дороги/парки/воду "занадто високими" і ламає пропорції.
                # Масштабуємо ОДНАКОВО по X/Y/Z
                print(f"[DEBUG] Before scaling: Z from {combined.bounds[0][2]:.2f} to {combined.bounds[1][2]:.2f}, scale={scale_factor:.6f}")
                s = trimesh.transformations.scale_matrix(scale_factor)
                combined.apply_transform(s)
                
                bounds_scaled = combined.bounds
                size_scaled = bounds_scaled[1] - bounds_scaled[0]
                print(f"Масштабування: {size_scaled[0]:.2f} x {size_scaled[1]:.2f} x {size_scaled[2]:.2f} мм")
                print(f"[DEBUG] After scaling: Z from {bounds_scaled[0][2]:.2f} to {bounds_scaled[1][2]:.2f}")
            # ВАЖЛИВО: не робимо примусового Z-scale (це спотворює висоти доріг/будівель
            # і дає ефект "дороги занадто високі" / "висять над землею").
        
        # Накопичуємо трансформації і застосовуємо їх і до об'єднаного, і до окремих мешів
        transforms: List[np.ndarray] = []

        if rotate_to_ground:
            try:
                angle_x = -np.pi / 2
                rot_x = trimesh.transformations.rotation_matrix(angle_x, [1, 0, 0])
                combined.apply_transform(rot_x)
                transforms.append(rot_x)
                print("Поворот навколо X застосовано (модель покладено на XY)")
            except Exception as e:
                print(f"Попередження: не вдалося покласти модель на XY (STL): {e}")

        try:
            final_bounds = combined.bounds
            final_center_from_bounds = (final_bounds[0] + final_bounds[1]) / 2.0
            if preserve_xy and preserve_z:
                t_center = np.eye(4)
            elif preserve_xy and not preserve_z:
                t_center = trimesh.transformations.translation_matrix([0.0, 0.0, -final_center_from_bounds[2]])
                combined.apply_translation([0.0, 0.0, -final_center_from_bounds[2]])
            elif preserve_z:
                t_center = trimesh.transformations.translation_matrix([-final_center_from_bounds[0], -final_center_from_bounds[1], 0.0])
                combined.apply_translation([-final_center_from_bounds[0], -final_center_from_bounds[1], 0.0])
            else:
                t_center = trimesh.transformations.translation_matrix(-final_center_from_bounds)
                combined.apply_translation(-final_center_from_bounds)
            if not (preserve_xy and preserve_z):
                transforms.append(t_center)

            if not preserve_z:
                final_bounds_after = combined.bounds
                min_z = final_bounds_after[0][2]
                t_minz = trimesh.transformations.translation_matrix([0, 0, -min_z])
                combined.apply_translation([0, 0, -min_z])
                transforms.append(t_minz)

            # Центруємо лише X/Y, щоб Z=0 залишився площиною друку
            if not preserve_xy:
                final_centroid_before = combined.centroid
                t_xy = trimesh.transformations.translation_matrix([-final_centroid_before[0], -final_centroid_before[1], 0.0])
                combined.apply_translation([-final_centroid_before[0], -final_centroid_before[1], 0.0])
                transforms.append(t_xy)

            final_bounds_check = combined.bounds
            final_centroid = combined.centroid
            print(f"Фінальні bounds: min={final_bounds_check[0]}, max={final_bounds_check[1]}")
            print(f"Фінальний центр (centroid): {final_centroid}")
        except Exception as e:
            print(f"Попередження: Центрування/вирівнювання не виконано: {e}")
        
        
        # Експортуємо об'єднаний (для STL одна геометрія)
        print(f"[DEBUG] Final mesh before STL export: {len(combined.vertices)}v, {len(combined.faces)}f")
        print(f"[DEBUG] Final bounds before STL export: Z from {combined.bounds[0][2]:.2f} to {combined.bounds[1][2]:.2f}")
        
        combined.export(filename, file_type="stl")
        print(f"Експортовано STL: {filename}")
        
        # Перевіряємо розмір файлу
        file_size = os.path.getsize(filename)
        print(f"Розмір файлу: {file_size} байт")
        if file_size < 84:  # Мінімальний розмір STL (80 байт заголовок + 4 байти кількість трикутників)
            raise ValueError(f"Файл занадто малий ({file_size} байт), можливо експорт не вдався")
        
        # Експортуємо окремі частини для preview (якщо потрібно)
        # Використовуємо transforms для кожного меша окремо
        outputs: dict[str, str] = {}
        part_map = {
            "Base": "base",
            "BaseFlat": "base",
            "Roads": "roads",
            "Buildings": "buildings",
            "Building": "buildings",  # Додано для Building_0, Building_1 тощо
            "Water": "water",
            "Parks": "parks",
            "POI": "poi",
        }
        
        
        try:
            # Групуємо меші по типах для об'єднання (наприклад, всі Building_* в один buildings)
            grouped_meshes: dict[str, list] = {}
            for name, mesh in working_items:
                # Перевіряємо, чи це частина, яку потрібно експортувати
                name_key = name.split("_")[0]  # Беремо першу частину для "Building_0" -> "Building"
                part_key = part_map.get(name_key) or part_map.get(name)
                
                if part_key:
                    if part_key not in grouped_meshes:
                        grouped_meshes[part_key] = []
                    grouped_meshes[part_key].append((name, mesh))
            
            # Експортуємо згруповані меші
            for part_key, meshes in grouped_meshes.items():
                try:
                    # Якщо кілька мешів одного типу - об'єднуємо їх
                    if len(meshes) == 1:
                        mesh_part = meshes[0][1].copy()  # meshes[0] = (name, mesh), тому meshes[0][1] = mesh
                    else:
                        # Об'єднуємо всі меші одного типу
                        # meshes - це список кортежів (name, mesh), тому беремо mesh (другий елемент)
                        mesh_parts = [mesh.copy() for _, mesh in meshes]
                        mesh_part = trimesh.util.concatenate(mesh_parts)
                    
                    # Застосовуємо всі трансформації до окремого меша
                    for t in transforms:
                        if isinstance(t, np.ndarray):
                            mesh_part.apply_transform(t)
                    
                    # Експортуємо частину
                    part_filename = filename.replace(".stl", f"_{part_key}.stl")
                    mesh_part.export(part_filename, file_type="stl")
                    outputs[part_key] = part_filename
                    print(f"Експортовано частину {part_key}: {part_filename} ({len(mesh_part.vertices)} вершин)")
                except Exception as e:
                    print(f"[WARN] Не вдалося експортувати частину {part_key}: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"[WARN] Експорт окремих частин не вдався: {e}")
            import traceback
            traceback.print_exc()
        
        return outputs
        
    except Exception as e:
        print(f"Помилка експорту STL: {e}")
        import traceback
        traceback.print_exc()
        raise
