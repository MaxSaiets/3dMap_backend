"""
TerrainProvider - клас для інтерполяції висот рельєфу
Дозволяє отримувати висоту землі в будь-якій точці (X, Y)
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Optional, Tuple


class TerrainSurfaceSampler:
    """
    Samples Z from an arbitrary terrain *surface mesh* (top surface) by interpolating
    within triangles in XY.

    This is used to drape overlays (roads/parks/water) onto the *actual* terrain mesh
    surface when the terrain was built via polygon-aware triangulation (stitching mode),
    where a heightfield-grid interpolator can diverge near boundaries/steep areas.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        v = np.asarray(vertices, dtype=float)
        f = np.asarray(faces, dtype=np.int64)
        if v.ndim != 2 or v.shape[1] < 3 or f.ndim != 2 or f.shape[1] != 3:
            raise ValueError("TerrainSurfaceSampler: invalid vertices/faces")
        if len(v) == 0 or len(f) == 0:
            raise ValueError("TerrainSurfaceSampler: empty mesh")

        self.v = v[:, :3].copy()
        self.f = f[:, :3].copy()

        # Precompute per-face XY bounds for spatial lookup
        tri = self.v[self.f]  # (F,3,3)
        xy = tri[:, :, :2]
        self.face_min = np.min(xy, axis=1)  # (F,2)
        self.face_max = np.max(xy, axis=1)  # (F,2)

        # Optional rtree index for fast candidate lookup
        self._rtree = None
        try:
            import importlib
            rtree_mod = importlib.import_module("rtree")
            index_mod = importlib.import_module("rtree.index")
            RTreeIndex = getattr(index_mod, "Index", None)
            if RTreeIndex is None:
                raise ImportError("rtree.index.Index not found")

            props = getattr(rtree_mod, "index").Property()
            props.dimension = 2
            idx = RTreeIndex(properties=props)  # type: ignore[misc]
            for fi in range(len(self.f)):
                mn = self.face_min[fi]
                mx = self.face_max[fi]
                # Skip degenerate bounds
                if not np.isfinite(mn).all() or not np.isfinite(mx).all():
                    continue
                if (mx[0] - mn[0]) < 1e-12 and (mx[1] - mn[1]) < 1e-12:
                    continue
                idx.insert(int(fi), (float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])))
            self._rtree = idx
        except Exception:
            self._rtree = None

    @staticmethod
    def _barycentric_z(pxy: np.ndarray, txy: np.ndarray, tz: np.ndarray, eps: float = 1e-10) -> Optional[float]:
        """
        Compute barycentric interpolation of z at point p inside triangle.
        pxy: (2,), txy: (3,2), tz: (3,)
        Returns z if inside (with epsilon), else None.
        """
        x, y = float(pxy[0]), float(pxy[1])
        x1, y1 = float(txy[0, 0]), float(txy[0, 1])
        x2, y2 = float(txy[1, 0]), float(txy[1, 1])
        x3, y3 = float(txy[2, 0]), float(txy[2, 1])

        det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(det) < eps:
            return None
        w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
        w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
        w3 = 1.0 - w1 - w2

        if (w1 >= -eps) and (w2 >= -eps) and (w3 >= -eps):
            return float(w1 * float(tz[0]) + w2 * float(tz[1]) + w3 * float(tz[2]))
        return None

    def sample(self, points_xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xy, dtype=float)
        if pts.ndim != 2 or pts.shape[1] < 2:
            return np.array([], dtype=float)
        pts = pts[:, :2]

        out = np.full((len(pts),), np.nan, dtype=float)
        tri = self.v[self.f]  # (F,3,3)

        for i in range(len(pts)):
            px, py = float(pts[i, 0]), float(pts[i, 1])
            if not np.isfinite(px) or not np.isfinite(py):
                continue

            # Candidate faces by bbox lookup
            cand = None
            if self._rtree is not None:
                try:
                    cand = list(self._rtree.intersection((px, py, px, py)))
                except Exception:
                    cand = None
            if not cand:
                # Fallback: brute-force scan a small number of nearest bbox faces
                # (still much cheaper than scanning all faces for typical sizes)
                dx = np.maximum(0.0, np.maximum(self.face_min[:, 0] - px, px - self.face_max[:, 0]))
                dy = np.maximum(0.0, np.maximum(self.face_min[:, 1] - py, py - self.face_max[:, 1]))
                d2 = dx * dx + dy * dy
                # Take a small batch of candidates
                k = min(64, len(d2))
                cand = np.argpartition(d2, k - 1)[:k].tolist() if k > 0 else []

            pxy = np.array([px, py], dtype=float)
            found = None
            for fi in cand:
                try:
                    t = tri[int(fi)]
                    txy = t[:, :2]
                    tz = t[:, 2]
                    z = self._barycentric_z(pxy, txy, tz, eps=1e-10)
                    if z is not None:
                        found = z
                        break
                except Exception:
                    continue
            if found is not None:
                out[i] = float(found)

        return out


class TerrainProvider:
    """
    Надає інтерполяцію висот рельєфу для будь-якої точки (X, Y)
    """
    
    def __init__(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, original_Z: Optional[np.ndarray] = None):
        """
        Ініціалізує TerrainProvider з сіткою висот
        
        Args:
            X: 2D масив X координат (meshgrid)
            Y: 2D масив Y координат (meshgrid)
            Z: 2D масив висот (meshgrid)
            original_Z: Опціональний 2D масив "оригінальних" висот (до модифікацій/депресії)
        """
        # Витягуємо 1D осі з meshgrid
        self.x_axis = X[0, :] if X.ndim == 2 else X
        self.y_axis = Y[:, 0] if Y.ndim == 2 else Y
        # Зберігаємо сітку висот (2D) — потрібна для інтерполяції
        self.z_grid = Z.astype(float, copy=False)


        # Зберігаємо мінімальну та максимальну висоту для fallback
        self.min_z = float(np.nanmin(Z)) if np.any(~np.isnan(Z)) else 0.0
        self.max_z = float(np.nanmax(Z)) if np.any(~np.isnan(Z)) else 0.0

        # Межі для клампу (щоб уникнути екстраполяції, яка часто "тягне" дороги/будівлі вниз/вгору)
        self.min_x = float(np.min(self.x_axis))
        self.max_x = float(np.max(self.x_axis))
        self.min_y = float(np.min(self.y_axis))
        self.max_y = float(np.max(self.y_axis))
        
        # Створюємо інтерполятор
        # RegularGridInterpolator очікує (y, x) порядок для осей
        self.interpolator = RegularGridInterpolator(
            (self.y_axis, self.x_axis),
            Z,
            bounds_error=False,
            # Критично: не екстраполюємо за межі (fill мінімальною висотою)
            fill_value=self.min_z,
            method='linear'
        )

        # Optional triangle-surface sampler (set by terrain_generator when available)
        self.surface_sampler: Optional[TerrainSurfaceSampler] = None
        
        # Provider для оригінальних висот (якщо задано)
        self.original_heights_provider = None
        if original_Z is not None:
            self.original_heights_provider = TerrainProvider(X, Y, original_Z)


    def _heights_on_terrain_triangles(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Інтерполяція висоти, яка ПОВНІСТЮ збігається з трикутниками terrain mesh.

        Важливо: terrain mesh будується з регулярної сітки і розбиває кожну клітинку
        на два трикутники по діагоналі між bottom_left та top_right (див. create_grid_faces).

        Це прибирає ефект "дороги в текстурі / в повітрі", який з'являється,
        коли draping робиться білінійною інтерполяцією, а рельєф — трикутниками.
        """
        xs = np.clip(xs.astype(float), self.min_x, self.max_x)
        ys = np.clip(ys.astype(float), self.min_y, self.max_y)

        # Індекси клітинки
        j = np.searchsorted(self.x_axis, xs, side="right") - 1
        i = np.searchsorted(self.y_axis, ys, side="right") - 1
        j = np.clip(j, 0, len(self.x_axis) - 2)
        i = np.clip(i, 0, len(self.y_axis) - 2)

        x0 = self.x_axis[j]
        x1 = self.x_axis[j + 1]
        y0 = self.y_axis[i]
        y1 = self.y_axis[i + 1]

        # Нормалізовані координати в межах клітинки [0..1]
        eps = 1e-12
        dx = (xs - x0) / (x1 - x0 + eps)
        dy = (ys - y0) / (y1 - y0 + eps)
        dx = np.clip(dx, 0.0, 1.0)
        dy = np.clip(dy, 0.0, 1.0)

        # Висоти 4-х кутів клітинки
        z00 = self.z_grid[i, j]         # top_left    (dx=0, dy=0)
        z10 = self.z_grid[i, j + 1]     # top_right   (dx=1, dy=0)
        z01 = self.z_grid[i + 1, j]     # bottom_left (dx=0, dy=1)
        z11 = self.z_grid[i + 1, j + 1] # bottom_right(dx=1, dy=1)

        # Трикутники, як у create_grid_faces:
        # T1: top_left (0,0), bottom_left (0,1), top_right (1,0)  => dx + dy <= 1
        # T2: top_right (1,0), bottom_left (0,1), bottom_right (1,1) => dx + dy > 1
        mask = (dx + dy) <= 1.0
        z = np.empty_like(dx, dtype=float)

        # Для T1: z = z00*(1-dx-dy) + z10*dx + z01*dy
        z[mask] = z00[mask] * (1.0 - dx[mask] - dy[mask]) + z10[mask] * dx[mask] + z01[mask] * dy[mask]

        # Для T2: ваги (w11=dx+dy-1, w10=1-dy, w01=1-dx), сума=1
        inv_mask = ~mask
        z[inv_mask] = (
            z11[inv_mask] * (dx[inv_mask] + dy[inv_mask] - 1.0)
            + z10[inv_mask] * (1.0 - dy[inv_mask])
            + z01[inv_mask] * (1.0 - dx[inv_mask])
        )

        # NaN -> min_z
        z = np.where(np.isnan(z), self.min_z, z)
        return z
    
    def get_height_at(self, x: float, y: float) -> float:
        """
        Отримує висоту землі в точці (x, y)
        
        Args:
            x: X координата (схід/захід, easting)
            y: Y координата (північ/південь, northing)
            
        Returns:
            Висота Z в точці (x, y), або мінімальна висота якщо точка за межами
        
        Примітка: RegularGridInterpolator очікує (y, x) порядок для осей,
        але координати передаються як (x, y) де x = схід/захід, y = північ/південь
        """
        try:
            z = self._heights_on_terrain_triangles(np.array([x]), np.array([y]))[0]
            return float(z) if not np.isnan(z) else self.min_z
        except Exception:
            # fallback: старий білінійний інтерполятор
            try:
                x = float(np.clip(x, self.min_x, self.max_x))
                y = float(np.clip(y, self.min_y, self.max_y))
                z = self.interpolator((y, x))
                if z is None or np.isnan(z):
                    return self.min_z
                return float(z)
            except Exception:
                return self.min_z
    
    def get_heights_for_points(self, points: np.ndarray) -> np.ndarray:
        """
        Отримує висоти для масиву точок
        
        Args:
            points: Масив форми (N, 2) з координатами [x, y]
            
        Returns:
            Масив висот форми (N,)
        """
        if len(points) == 0:
            return np.array([])
        
        try:
            xs = points[:, 0].astype(float, copy=False)
            ys = points[:, 1].astype(float, copy=False)
            heights = self._heights_on_terrain_triangles(xs, ys)
            return heights
        except Exception:
            # fallback: старий білінійний інтерполятор
            try:
                xs = np.clip(points[:, 0].astype(float), self.min_x, self.max_x)
                ys = np.clip(points[:, 1].astype(float), self.min_y, self.max_y)
                yx_points = np.column_stack([ys, xs])
                heights = self.interpolator(yx_points)
                heights = np.where(np.isnan(heights), self.min_z, heights)
                return heights
            except Exception:
                return np.full(len(points), self.min_z)

    def get_surface_heights_for_points(self, points: np.ndarray) -> np.ndarray:
        """
        Returns heights sampled from the final terrain *surface mesh* when available.
        Falls back to the regular heightfield interpolation otherwise.
        """
        if points is None:
            return np.array([], dtype=float)
        pts = np.asarray(points)
        if pts.ndim != 2 or pts.shape[1] < 2:
            return np.array([], dtype=float)

        sampler = getattr(self, "surface_sampler", None)
        if sampler is None:
            return self.get_heights_for_points(pts)

        try:
            z = sampler.sample(pts[:, :2])
            # Fill any NaN gaps with heightfield interpolation (robust)
            if z is None or len(z) != len(pts):
                return self.get_heights_for_points(pts)
            miss = ~np.isfinite(z)
            if np.any(miss):
                z2 = self.get_heights_for_points(pts[miss])
                z = np.asarray(z, dtype=float)
                z[miss] = np.asarray(z2, dtype=float).reshape(-1)
            return np.asarray(z, dtype=float)
        except Exception:
            return self.get_heights_for_points(pts)
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """
        Повертає межі рельєфу (min_x, max_x, min_y, max_y)
        """
        return (
            float(np.min(self.x_axis)),
            float(np.max(self.x_axis)),
            float(np.min(self.y_axis)),
            float(np.max(self.y_axis))
        )

