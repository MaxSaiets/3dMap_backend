"""
Сервіс для перевірки та покращення якості mesh для 3D принтера
Перевіряє мінімальні розміри, товщини, watertight та інші параметри
"""
import trimesh
import numpy as np
from typing import Optional, Tuple, List


# Мінімальні розміри для 3D принтера (в міліметрах після масштабування)
MIN_WALL_THICKNESS_MM = 0.3  # Мінімальна товщина стінки
MIN_FEATURE_SIZE_MM = 0.2    # Мінімальний розмір деталі
MIN_OVERHANG_ANGLE = 45.0     # Мінімальний кут для overhang (градуси)


def validate_mesh_for_3d_printing(
    mesh: trimesh.Trimesh,
    scale_factor: Optional[float] = None,
    model_size_mm: float = 100.0,
) -> Tuple[bool, List[str]]:
    """
    Перевіряє mesh на придатність для 3D принтера
    
    Args:
        mesh: Trimesh об'єкт для перевірки
        scale_factor: Фактор масштабування (якщо відомий)
        model_size_mm: Розмір моделі в мм (для оцінки масштабу)
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_warnings)
    """
    warnings = []
    
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return False, ["Mesh порожній або невалідний"]
    
    # 1. Перевірка watertight
    if not mesh.is_watertight:
        warnings.append("Mesh не є watertight (можуть бути дірки)")
        try:
            mesh.fill_holes()
            if not mesh.is_watertight:
                warnings.append("Не вдалося заповнити всі дірки")
        except Exception as e:
            warnings.append(f"Помилка заповнення дірок: {e}")
    
    # 2. Перевірка мінімальних розмірів
    try:
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        min_dim = float(np.min(size))
        
        # Оцінюємо масштаб
        if scale_factor is None:
            # Оцінюємо за model_size_mm
            avg_xy = (size[0] + size[1]) / 2.0
            if avg_xy > 0:
                estimated_scale = model_size_mm / avg_xy
            else:
                estimated_scale = 1.0
        else:
            estimated_scale = scale_factor
        
        # Конвертуємо в мм
        min_dim_mm = min_dim * estimated_scale
        
        if min_dim_mm < MIN_FEATURE_SIZE_MM:
            warnings.append(f"Мінімальний розмір занадто малий: {min_dim_mm:.2f}мм (мінімум: {MIN_FEATURE_SIZE_MM}мм)")
    except Exception as e:
        warnings.append(f"Помилка перевірки розмірів: {e}")
    
    # 3. Перевірка товщини стінок
    try:
        # Оцінюємо товщину через аналіз відстаней між поверхнями
        # Це спрощена перевірка - для точної потрібен більш складний аналіз
        if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
            # Перевіряємо чи є дуже тонкі частини через аналіз граней
            # Обчислюємо площі граней вручну
            face_areas = []
            for face in mesh.faces:
                v0, v1, v2 = mesh.vertices[face]
                area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                face_areas.append(area)
            
            if len(face_areas) > 0:
                min_area = float(np.min(face_areas))
                min_area_mm2 = min_area * (estimated_scale ** 2)
                if min_area_mm2 < 0.01:  # Дуже мала грань
                    # Avoid "²" which may crash on some Windows console encodings.
                    warnings.append(f"Знайдено дуже малі грані: {min_area_mm2:.4f}мм^2")
    except Exception as e:
        warnings.append(f"Помилка перевірки товщини: {e}")
    
    # 4. Перевірка нормалей
    try:
        if not mesh.is_winding_consistent:
            warnings.append("Порядок вершин граней неконсистентний")
    except:
        pass
    
    # 5. Перевірка дегенерованих граней
    try:
        # Перевіряємо площу граней
        if len(mesh.faces) > 0:
            face_areas = []
            for face in mesh.faces:
                v0, v1, v2 = mesh.vertices[face]
                area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                face_areas.append(area)
            
            if len(face_areas) > 0:
                face_areas = np.array(face_areas)
                degenerate_count = np.sum(face_areas < 1e-10)
                if degenerate_count > 0:
                    warnings.append(f"Знайдено {degenerate_count} дегенерованих граней")
    except:
        pass
    
    is_valid = len(warnings) == 0 or all("можуть бути" in w or "дуже малі" in w for w in warnings)
    return is_valid, warnings


def improve_mesh_for_3d_printing(
    mesh: trimesh.Trimesh,
    aggressive: bool = False,
) -> trimesh.Trimesh:
    """
    Покращує mesh для 3D принтера
    
    Args:
        mesh: Trimesh об'єкт для покращення
        aggressive: Якщо True, застосовує більш агресивні виправлення
        
    Returns:
        Покращений Trimesh об'єкт
    """
    if mesh is None:
        return None
    
    improved = mesh.copy()
    
    try:
        # 1. Виправлення нормалей
        improved.fix_normals()
        
        # 2. Видалення дегенерованих граней
        improved.remove_duplicate_faces()
        improved.remove_unreferenced_vertices()
        
        # 3. Заповнення дірок
        if not improved.is_watertight:
            improved.fill_holes()
        
        # 4. Об'єднання близьких вершин
        improved.merge_vertices(merge_tex=True, merge_norm=True)
        
        # 5. Агресивні виправлення
        if aggressive:
            try:
                # Виправлення порядку вершин
                if not improved.is_winding_consistent:
                    trimesh.repair.fix_winding(improved)
                # Додаткове очищення
                improved.remove_duplicate_faces()
                improved.remove_unreferenced_vertices()
            except Exception as e:
                print(f"[WARN] Помилка агресивних виправлень: {e}")
        
        # 6. Фінальна перевірка
        if len(improved.vertices) == 0 or len(improved.faces) == 0:
            print("[WARN] Mesh став порожнім після покращень, повертаємо оригінал")
            return mesh
        
        return improved
        
    except Exception as e:
        print(f"[WARN] Помилка покращення mesh: {e}")
        return mesh


def check_minimum_thickness(
    mesh: trimesh.Trimesh,
    scale_factor: Optional[float] = None,
    model_size_mm: float = 100.0,
) -> Tuple[bool, float]:
    """
    Перевіряє мінімальну товщину mesh
    
    Returns:
        Tuple[bool, float]: (is_valid, min_thickness_mm)
    """
    try:
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        min_dim = float(np.min(size))
        
        # Оцінюємо масштаб
        if scale_factor is None:
            avg_xy = (size[0] + size[1]) / 2.0
            if avg_xy > 0:
                estimated_scale = model_size_mm / avg_xy
            else:
                estimated_scale = 1.0
        else:
            estimated_scale = scale_factor
        
        min_thickness_mm = min_dim * estimated_scale
        is_valid = min_thickness_mm >= MIN_WALL_THICKNESS_MM
        
        return is_valid, min_thickness_mm
    except Exception:
        return False, 0.0

