import json
from pathlib import Path

from services.crs_utils import bbox_latlon_to_utm
from services.global_center import get_or_create_global_center


def test_regression_case_local_bbox_matches_expected_if_present():
    """
    Fast, offline check: verifies that a regression case bbox produces expected local ranges.
    This catches accidental changes in GlobalCenter/bbox conversion and gives a stable anchor for debugging.
    """
    cases_dir = Path(__file__).resolve().parents[1] / "regression" / "cases"
    for cf in sorted(cases_dir.glob("*.json")):
        case = json.loads(cf.read_text(encoding="utf-8"))
        req = case.get("request") or {}
        expected = case.get("expected_local_bbox_m") or {}
        tol = float(expected.get("tolerance_m", 0.0) or 0.0)
        if not expected or tol <= 0:
            continue

        north = float(req["north"])
        south = float(req["south"])
        east = float(req["east"])
        west = float(req["west"])
        gc = get_or_create_global_center(bbox_latlon=(north, south, east, west))

        minx_utm, miny_utm, maxx_utm, maxy_utm = bbox_latlon_to_utm(north, south, east, west)[:4]
        minx_local, miny_local = gc.to_local(minx_utm, miny_utm)
        maxx_local, maxy_local = gc.to_local(maxx_utm, maxy_utm)

        assert abs(float(minx_local) - float(expected["x_min"])) <= tol
        assert abs(float(maxx_local) - float(expected["x_max"])) <= tol
        assert abs(float(miny_local) - float(expected["y_min"])) <= tol
        assert abs(float(maxy_local) - float(expected["y_max"])) <= tol


