import numpy as np
import pytest


def _has_rasterio() -> bool:
    try:
        import rasterio  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_rasterio(), reason="rasterio not available")
def test_flatten_heightfield_under_buildings_makes_constant_under_mask():
    from shapely.geometry import Polygon
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    from services.terrain_generator import flatten_heightfield_under_buildings

    # Regular grid
    rows, cols = 40, 50
    x = np.linspace(0.0, 100.0, cols)
    y = np.linspace(0.0, 80.0, rows)
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Terrain with slope + small noise
    Z = 0.02 * X + 0.01 * Y
    Z = Z + 0.05 * np.sin(X / 7.0) * np.cos(Y / 9.0)

    # A "building" polygon near center
    poly = Polygon([(30, 20), (70, 20), (70, 60), (30, 60)])

    Z2 = flatten_heightfield_under_buildings(X=X, Y=Y, Z=Z, building_geometries=[poly], quantile=0.90)

    transform = from_bounds(float(np.min(X)), float(np.min(Y)), float(np.max(X)), float(np.max(Y)), cols, rows)
    mask = rasterize([(poly, 1)], out_shape=(rows, cols), transform=transform, fill=0, dtype="uint8", all_touched=True).astype(bool)
    assert int(mask.sum()) > 5

    # Under the building: all heights should become the same (flattened)
    under = Z2[mask]
    assert np.all(np.isfinite(under))
    assert float(np.max(under) - np.min(under)) < 1e-9


@pytest.mark.skipif(not _has_rasterio(), reason="rasterio not available")
def test_flatten_heightfield_under_buildings_only_affects_inside():
    from shapely.geometry import Polygon
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    from services.terrain_generator import flatten_heightfield_under_buildings

    rows, cols = 30, 30
    x = np.linspace(0.0, 30.0, cols)
    y = np.linspace(0.0, 30.0, rows)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = X + 0.5 * Y

    poly = Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])
    Z2 = flatten_heightfield_under_buildings(X=X, Y=Y, Z=Z, building_geometries=[poly], quantile=0.90)

    transform = from_bounds(float(np.min(X)), float(np.min(Y)), float(np.max(X)), float(np.max(Y)), cols, rows)
    mask = rasterize([(poly, 1)], out_shape=(rows, cols), transform=transform, fill=0, dtype="uint8", all_touched=True).astype(bool)

    # Outside should remain the same (exactly, since Z is deterministic and flatten only writes inside)
    assert np.allclose(Z2[~mask], Z[~mask])


