import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from services.footprints_loader import transfer_osm_attributes_to_footprints


def test_transfer_osm_attributes_to_footprints_prefers_overlap():
    # Footprint square
    fp = gpd.GeoDataFrame(
        {"geometry": [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])]},
        crs="EPSG:3857",
    )

    # Two OSM buildings: one small overlap, one large overlap
    osm = gpd.GeoDataFrame(
        {
            "height": [5.0, 20.0],
            "geometry": [
                Polygon([(9, 9), (12, 9), (12, 12), (9, 12)]),  # tiny overlap
                Polygon([(1, 1), (9, 1), (9, 9), (1, 9)]),      # big overlap
            ],
        },
        crs="EPSG:3857",
    )

    out = transfer_osm_attributes_to_footprints(fp, osm)
    assert "height" in out.columns
    assert np.isclose(float(out.loc[0, "height"]), 20.0)


