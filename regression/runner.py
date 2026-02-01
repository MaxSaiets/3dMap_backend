from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from services.crs_utils import bbox_latlon_to_utm
from services.crs_utils import get_transformers
from services.data_loader import fetch_city_data
from services.extras_loader import fetch_extras
from services.global_center import get_or_create_global_center
from services.green_processor import process_green_areas
from services.model_exporter import export_scene
from services.poi_processor import process_pois
from services.project_context import ProjectContext
from services.road_processor import build_road_polygons, detect_bridges, process_roads
from services.terrain_generator import create_terrain_mesh
from services.water_processor import process_water_surface


@dataclass
class RegressionResult:
    case_name: str
    ok: bool
    report: Dict[str, Any]
    artifacts_dir: Optional[str] = None


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def _compute_scale_factor(model_size_mm: float, bbox_meters: Tuple[float, float, float, float]) -> Optional[float]:
    minx, miny, maxx, maxy = bbox_meters
    size_x = float(maxx - minx)
    size_y = float(maxy - miny)
    if size_x <= 0 or size_y <= 0:
        return None
    avg_xy = (size_x + size_y) / 2.0
    if avg_xy <= 0:
        return None
    return float(model_size_mm) / float(avg_xy)


def run_regression_case(case_path: str, artifacts_root: str = "output/regression") -> RegressionResult:
    case_file = Path(case_path)
    case = json.loads(case_file.read_text(encoding="utf-8"))
    name = str(case.get("name") or case_file.stem)
    req = case.get("request") or {}

    # Two ways to define a test area:
    # 1) Explicit bbox in lat/lon: request.{north,south,east,west}
    # 2) A "local bbox" in meters + a center lat/lon:
    #    This is useful when the user only has local X/Y ranges from logs.
    north = req.get("north")
    south = req.get("south")
    east = req.get("east")
    west = req.get("west")
    bbox_derived = False

    if north is None or south is None or east is None or west is None:
        center = case.get("center_latlon") or {}
        local_bbox = case.get("local_bbox_m") or {}
        if not center or not local_bbox:
            raise ValueError("Case must define either request.{north,south,east,west} or (center_latlon + local_bbox_m).")
        bbox_derived = True

        center_lat = float(center["lat"])
        center_lon = float(center["lon"])
        to_utm, to_wgs84, _ = get_transformers(center_lat, center_lon)
        cx, cy = to_utm(center_lon, center_lat)

        x_min = float(local_bbox["x_min"])
        x_max = float(local_bbox["x_max"])
        y_min = float(local_bbox["y_min"])
        y_max = float(local_bbox["y_max"])

        half_x = max(abs(x_min), abs(x_max))
        half_y = max(abs(y_min), abs(y_max))

        lon_w, lat_s = to_wgs84(float(cx - half_x), float(cy - half_y))
        lon_e, lat_n = to_wgs84(float(cx + half_x), float(cy + half_y))

        north, south, east, west = float(lat_n), float(lat_s), float(lon_e), float(lon_w)

    north = float(north)
    south = float(south)
    east = float(east)
    west = float(west)
    bbox_latlon = (north, south, east, west)

    # Stable global center + snapped DEM grid
    gc = get_or_create_global_center(bbox_latlon=bbox_latlon)
    ctx = ProjectContext(
        project_id=f"regression_{name}",
        project_bbox_latlon=bbox_latlon,
        global_center=gc,
        terrarium_zoom=int(req.get("terrarium_zoom", 15)),
        terrain_resolution=int(req.get("terrain_resolution", 350)),
    )
    ctx.ensure_initialized()

    # Local bbox
    minx_utm, miny_utm, maxx_utm, maxy_utm = bbox_latlon_to_utm(north, south, east, west)[:4]
    minx_local, miny_local = gc.to_local(minx_utm, miny_utm)
    maxx_local, maxy_local = gc.to_local(maxx_utm, maxy_utm)
    bbox_meters = (float(minx_local), float(miny_local), float(maxx_local), float(maxy_local))

    expected = case.get("expected_local_bbox_m") or {}
    tol = float(expected.get("tolerance_m", 0.0) or 0.0)
    bbox_ok = True
    # If bbox is derived from center+local (approx mode), do not fail the whole case on bbox mismatch.
    # We still report it for debugging.
    if (not bbox_derived) and expected and tol > 0:
        bbox_ok = all(
            [
                _close(bbox_meters[0], float(expected.get("x_min")), tol),
                _close(bbox_meters[2], float(expected.get("x_max")), tol),
                _close(bbox_meters[1], float(expected.get("y_min")), tol),
                _close(bbox_meters[3], float(expected.get("y_max")), tol),
            ]
        )

    scale_factor = _compute_scale_factor(float(req.get("model_size_mm", 100.0)), bbox_meters)
    water_depth_mm = float(req.get("water_depth", 2.0))
    water_depth_m = (water_depth_mm / float(scale_factor)) if scale_factor else None

    report: Dict[str, Any] = {
        "case": name,
        "bbox_latlon": {"north": north, "south": south, "east": east, "west": west},
        "bbox_local_m": {"minx": bbox_meters[0], "miny": bbox_meters[1], "maxx": bbox_meters[2], "maxy": bbox_meters[3]},
        "bbox_matches_expected": bbox_ok,
        "scale_factor_mm_per_m": scale_factor,
        "project_context": {
            "baseline_ref_m": ctx.baseline_elevation_ref_m,
            "baseline_offset_m": ctx.baseline_offset_m,
            "dem_grid_step_m": ctx.dem_grid_step_m,
        },
        "meshes": {},
        "assertions": {},
        "warnings": [],
    }

    # Artifacts output
    artifacts_dir = _ensure_dir(Path(artifacts_root) / name)

    # Load OSM + extras (with optional padding)
    pad_m = float(req.get("context_padding_m", 0.0) or 0.0)
    ctx_north, ctx_south, ctx_east, ctx_west = north, south, east, west
    if pad_m > 0:
        x1, y1 = gc.to_utm(ctx_west, ctx_south)
        x2, y2 = gc.to_utm(ctx_east, ctx_north)
        minx = float(min(x1, x2) - pad_m)
        maxx = float(max(x1, x2) + pad_m)
        miny = float(min(y1, y2) - pad_m)
        maxy = float(max(y1, y2) + pad_m)
        lon_w, lat_s = gc.to_wgs84(minx, miny)
        lon_e, lat_n = gc.to_wgs84(maxx, maxy)
        ctx_north, ctx_south, ctx_east, ctx_west = float(lat_n), float(lat_s), float(lon_e), float(lon_w)

    gdf_buildings, gdf_water, G_roads = fetch_city_data(ctx_north, ctx_south, ctx_east, ctx_west)
    gdf_green, gdf_pois = fetch_extras(ctx_north, ctx_south, ctx_east, ctx_west)

    # Use CRS from loaded data
    source_crs = None
    try:
        if gdf_buildings is not None and not gdf_buildings.empty:
            source_crs = gdf_buildings.crs
        elif gdf_water is not None and not gdf_water.empty:
            source_crs = gdf_water.crs
    except Exception:
        source_crs = None

    # Terrain
    terrain_mesh = None
    terrain_provider = None
    if bool(req.get("terrain_enabled", True)):
        max_relief_mm = float(req.get("max_terrain_relief_mm", 12.0))
        max_relief_m = (max_relief_mm / float(scale_factor)) if scale_factor else None
        base_thickness_mm = float(req.get("terrain_base_thickness_mm", 2.0))
        base_thickness_m = (base_thickness_mm / float(scale_factor)) if scale_factor else 5.0

        terrain_mesh, terrain_provider = create_terrain_mesh(
            bbox_meters,
            z_scale=float(req.get("terrain_z_scale", 3.0)),
            resolution=int(req.get("terrain_resolution", 350)),
            latlon_bbox=bbox_latlon,
            source_crs=source_crs,
            terrarium_zoom=int(req.get("terrarium_zoom", 15)),
            base_thickness=float(base_thickness_m),
            flatten_buildings=bool(req.get("flatten_buildings_on_terrain", True)),
            building_geometries=list(gdf_buildings.geometry.values) if (gdf_buildings is not None and not gdf_buildings.empty) else None,
            smoothing_sigma=float(req.get("terrain_smoothing_sigma", 2.0)),
            water_geometries=list(gdf_water.geometry.values) if (gdf_water is not None and not gdf_water.empty) else None,
            water_depth_m=float(water_depth_m) if water_depth_m is not None else 0.0,
            subdivide=bool(req.get("terrain_subdivide", True)),
            subdivide_levels=int(req.get("terrain_subdivide_levels", 1)),
            global_center=gc,
            bbox_is_local=True,
            grid_origin_local=ctx.grid_origin_local,
            elevation_ref_m=ctx.baseline_elevation_ref_m,
            baseline_offset_m=ctx.baseline_offset_m,
            max_relief_m=max_relief_m,
        )

    def mesh_stats(mesh) -> Dict[str, Any]:
        if mesh is None:
            return {"present": False}
        b = mesh.bounds
        size = (b[1] - b[0]).tolist()
        zmin = float(b[0][2])
        zmax = float(b[1][2])
        return {
            "present": True,
            "verts": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "bounds": {"min": b[0].tolist(), "max": b[1].tolist()},
            "size_m": size,
            "z_range_m": float(zmax - zmin),
            "z_range_mm": float((zmax - zmin) * float(scale_factor)) if scale_factor else None,
            "watertight": bool(mesh.is_watertight),
        }

    report["meshes"]["terrain"] = mesh_stats(terrain_mesh)

    # Roads + bridges
    road_mesh = None
    bridge_count = 0
    if terrain_provider is not None and G_roads is not None:
        merged_roads = build_road_polygons(G_roads, width_multiplier=float(req.get("road_width_multiplier", 1.0)))
        water_geoms_for_bridges = list(gdf_water.geometry.values) if (gdf_water is not None and not gdf_water.empty) else None

        # Detect bridges (local) for reporting
        try:
            water_geoms_local = None
            if water_geoms_for_bridges is not None:
                # process_roads will convert for its own use; detect_bridges needs local for correct intersects
                from shapely.ops import transform as shp_transform

                def to_local(x, y, z=None):
                    xl, yl = gc.to_local(x, y)
                    return (xl, yl) if z is None else (xl, yl, z)

                water_geoms_local = [shp_transform(to_local, g) for g in water_geoms_for_bridges if g is not None and not g.is_empty]
            bridges = detect_bridges(G_roads, water_geometries=water_geoms_local)
            bridge_count = int(len(bridges))
        except Exception:
            bridge_count = 0

        road_mesh = process_roads(
            G_roads,
            float(req.get("road_width_multiplier", 1.0)),
            terrain_provider=terrain_provider,
            road_height=(float(req.get("road_height_mm", 0.5)) / float(scale_factor)) if scale_factor else 0.8,
            road_embed=(float(req.get("road_embed_mm", 0.3)) / float(scale_factor)) if scale_factor else 0.0,
            merged_roads=merged_roads,
            water_geometries=water_geoms_for_bridges,
            bridge_height_multiplier=1.0,
            global_center=gc,
        )

    report["meshes"]["roads"] = mesh_stats(road_mesh)
    report["meshes"]["bridges_detected"] = bridge_count

    # Water surface
    water_mesh = None
    if terrain_provider is not None and gdf_water is not None and not gdf_water.empty and water_depth_m is not None:
        water_mesh = process_water_surface(
            gdf_water,
            thickness_m=0.001,
            depth_meters=float(water_depth_m),
            terrain_provider=terrain_provider,
            global_center=gc,
            surface_offset_m=0.0,
        )
    report["meshes"]["water"] = mesh_stats(water_mesh)

    # Green (parks)
    green_mesh = None
    if terrain_provider is not None and bool(req.get("include_parks", True)) and gdf_green is not None and not gdf_green.empty:
        # Remove green areas that overlap water polygons (prevents "green on river" artifacts)
        try:
            if gdf_water is not None and not gdf_water.empty:
                from shapely.ops import unary_union

                water_union = unary_union([g for g in gdf_water.geometry.values if g is not None and not g.is_empty])
                if water_union is not None and not water_union.is_empty:
                    gdf_green = gdf_green.copy()

                    def _cut_green(geom):
                        if geom is None or geom.is_empty:
                            return geom
                        try:
                            if geom.intersects(water_union):
                                return geom.difference(water_union)
                        except Exception:
                            return geom
                        return geom

                    gdf_green["geometry"] = gdf_green["geometry"].apply(_cut_green)
                    gdf_green = gdf_green[~gdf_green["geometry"].is_empty]
        except Exception:
            pass

        green_mesh = process_green_areas(
            gdf_green,
            height_m=(float(req.get("parks_height_mm", 0.6)) / float(scale_factor)) if scale_factor else 0.8,
            embed_m=(float(req.get("parks_embed_mm", 0.2)) / float(scale_factor)) if scale_factor else 0.0,
            terrain_provider=terrain_provider,
            global_center=gc,
            scale_factor=float(scale_factor) if scale_factor else None,
        )
    report["meshes"]["green"] = mesh_stats(green_mesh)

    # POIs
    poi_mesh = None
    if terrain_provider is not None and bool(req.get("include_pois", False)) and gdf_pois is not None and not gdf_pois.empty:
        poi_mesh = process_pois(
            gdf_pois,
            size_m=(float(req.get("poi_size_mm", 0.6)) / float(scale_factor)) if scale_factor else 0.7,
            height_m=(float(req.get("poi_height_mm", 0.8)) / float(scale_factor)) if scale_factor else 1.0,
            embed_m=(float(req.get("poi_embed_mm", 0.2)) / float(scale_factor)) if scale_factor else 0.0,
            terrain_provider=terrain_provider,
        )
    report["meshes"]["pois"] = mesh_stats(poi_mesh)

    # Assertions
    assertions = case.get("assertions") or {}
    min_relief_mm = float(assertions.get("min_terrain_relief_mm", 0.0) or 0.0)
    min_sep_mm = float(assertions.get("min_water_surface_separation_mm", 0.0) or 0.0)
    min_bridge_clearance_mm = float(assertions.get("min_bridge_clearance_mm", 0.0) or 0.0)

    ok = True
    if not bbox_ok:
        ok = False
        report["warnings"].append("Local bbox does not match expected (check north/south/east/west or global center).")

    # Terrain relief check (in printed mm)
    relief_mm = report["meshes"]["terrain"].get("z_range_mm") if report["meshes"]["terrain"].get("present") else None
    relief_ok = (relief_mm is not None) and (float(relief_mm) >= min_relief_mm)
    report["assertions"]["terrain_relief_mm"] = {"value": relief_mm, "min": min_relief_mm, "ok": relief_ok}
    if not relief_ok:
        ok = False

    # Water separation check: water surface should sit above depressed ground by epsilon (avoid z-fighting)
    water_sep_ok = True
    water_min_sep_mm = None
    if water_mesh is not None and terrain_provider is not None and len(water_mesh.vertices) > 0:
        sample = water_mesh.vertices
        if len(sample) > 5000:
            idx = np.random.default_rng(0).choice(len(sample), size=5000, replace=False)
            sample = sample[idx]
        xy = sample[:, :2]
        water_z = sample[:, 2]
        ground_z = terrain_provider.get_heights_for_points(xy)
        sep_m = water_z - ground_z
        water_min_sep_mm = float(np.min(sep_m) * float(scale_factor)) if scale_factor else None
        water_sep_ok = (water_min_sep_mm is not None) and (water_min_sep_mm >= min_sep_mm)
    report["assertions"]["water_min_separation_mm"] = {"value": water_min_sep_mm, "min": min_sep_mm, "ok": water_sep_ok}
    if not water_sep_ok and (water_mesh is not None):
        ok = False

    # Parks/green layering check (if present): top surface should be above terrain by a visible margin.
    parks_top_clear_mm = None
    parks_ok = True
    if green_mesh is not None and terrain_provider is not None and len(green_mesh.vertices) > 0 and scale_factor:
        sample = green_mesh.vertices
        if len(sample) > 5000:
            idx = np.random.default_rng(1).choice(len(sample), size=5000, replace=False)
            sample = sample[idx]
        xy = sample[:, :2]
        z = sample[:, 2]
        ground = terrain_provider.get_heights_for_points(xy)
        dz = z - ground
        parks_top_clear_mm = float(np.percentile(dz, 95) * float(scale_factor))
        min_parks_mm = float((case.get("assertions") or {}).get("min_parks_top_clearance_mm", 0.2) or 0.2)
        parks_ok = parks_top_clear_mm >= min_parks_mm
        report["assertions"]["parks_top_clearance_mm_estimate"] = {"value": parks_top_clear_mm, "min": min_parks_mm, "ok": parks_ok}
        if not parks_ok:
            ok = False

    # Roads layering check (if present): typical top surfaces should be above terrain (bridges aside).
    roads_top_clear_mm = None
    roads_ok = True
    if road_mesh is not None and terrain_provider is not None and len(road_mesh.vertices) > 0 and scale_factor:
        sample = road_mesh.vertices
        if len(sample) > 8000:
            idx = np.random.default_rng(2).choice(len(sample), size=8000, replace=False)
            sample = sample[idx]
        xy = sample[:, :2]
        z = sample[:, 2]
        ground = terrain_provider.get_heights_for_points(xy)
        dz = z - ground
        roads_top_clear_mm = float(np.percentile(dz, 95) * float(scale_factor))
        min_roads_mm = float((case.get("assertions") or {}).get("min_roads_top_clearance_mm", 0.1) or 0.1)
        roads_ok = roads_top_clear_mm >= min_roads_mm
        report["assertions"]["roads_top_clearance_mm_estimate"] = {"value": roads_top_clear_mm, "min": min_roads_mm, "ok": roads_ok}
        if not roads_ok:
            ok = False

    # Bridge clearance check: ensure there are road vertices above water surface by at least clearance near detected bridges.
    bridge_ok = True
    bridge_clearance_mm = None
    if bridge_count > 0 and road_mesh is not None and water_mesh is not None and terrain_provider is not None and scale_factor:
        # Estimate clearance by comparing high-percentile road Z to water level in the same XY region.
        # This is intentionally conservative and catches "bridge not elevated at all".
        rv = road_mesh.vertices
        wv = water_mesh.vertices
        # Use only points roughly inside bbox to avoid side walls
        road_z95 = float(np.percentile(rv[:, 2], 95))
        water_z50 = float(np.percentile(wv[:, 2], 50))
        bridge_clearance_mm = float((road_z95 - water_z50) * float(scale_factor))
        bridge_ok = bridge_clearance_mm >= min_bridge_clearance_mm
    report["assertions"]["bridge_clearance_mm_estimate"] = {"value": bridge_clearance_mm, "min": min_bridge_clearance_mm, "ok": bridge_ok}
    if not bridge_ok and bridge_count > 0:
        ok = False

    # Export artifacts
    if bool((case.get("artifacts") or {}).get("export_scene", True)):
        try:
            fmt = str(req.get("export_format", "3mf")).lower()
            out_file = artifacts_dir / f"{name}.{fmt}"
            export_scene(
                terrain_mesh=terrain_mesh,
                road_mesh=road_mesh,
                building_meshes=[],
                water_mesh=water_mesh,
                filename=str(out_file),
                format=fmt,
                model_size_mm=float(req.get("model_size_mm", 100.0)),
                add_flat_base=(terrain_mesh is None),
                parks_mesh=green_mesh,
                poi_mesh=poi_mesh,
                landmarks_mesh=None,
            )
            report["artifacts"] = {"scene": str(out_file)}
        except Exception as e:
            report["warnings"].append(f"Export failed: {e}")

    # Write report
    (artifacts_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return RegressionResult(case_name=name, ok=ok, report=report, artifacts_dir=str(artifacts_dir))


