"""
Prepare a GeoPackage for fast bbox querying of Building Footprints.

Usage examples:
  python scripts/prepare_footprints_gpkg.py --input H:\\downloads\\ukraine.geojson --output cache\\footprints\\ukraine_buildings.gpkg
  python scripts/prepare_footprints_gpkg.py --input H:\\downloads\\ukraine.csv.gz --output cache\\footprints\\ukraine_buildings.gpkg

Supported inputs:
- .geojson / .json / .gpkg / .parquet (geopandas-readable)
- .geojson.gz
- .csv / .csv.gz with a WKT geometry column named 'geometry' or 'wkt' or 'geom'

Output:
- a single GeoPackage with layer name 'buildings' in EPSG:4326
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely import wkt


def _read_csv_with_wkt(path: Path) -> gpd.GeoDataFrame:
    if path.suffix.lower() == ".gz":
        df = pd.read_csv(path, compression="gzip")
    else:
        df = pd.read_csv(path)

    geom_col = None
    for c in ["geometry", "wkt", "geom"]:
        if c in df.columns:
            geom_col = c
            break
    if geom_col is None:
        raise ValueError("CSV must contain a WKT geometry column named one of: geometry|wkt|geom")

    geoms = df[geom_col].astype(str).map(wkt.loads)
    df = df.drop(columns=[geom_col])
    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")
    return gdf


def _read_geojson_gz(path: Path) -> gpd.GeoDataFrame:
    # geopandas can read gz via file-like, but on Windows it's more reliable to decompress to memory
    with gzip.open(path, "rb") as f:
        data = f.read()
    # write temp alongside
    tmp = path.with_suffix("")  # removes .gz
    tmp.write_bytes(data)
    try:
        return gpd.read_file(tmp)
    finally:
        try:
            tmp.unlink(missing_ok=True)  # py3.11
        except Exception:
            pass


def read_any(input_path: Path) -> gpd.GeoDataFrame:
    suffix = "".join(input_path.suffixes).lower()
    if suffix.endswith(".csv") or suffix.endswith(".csv.gz"):
        return _read_csv_with_wkt(input_path)
    if suffix.endswith(".geojson.gz") or suffix.endswith(".json.gz"):
        return _read_geojson_gz(input_path)
    return gpd.read_file(input_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input footprints file (geojson/csv/gpkg/parquet)")
    ap.add_argument("--output", required=True, help="Output GeoPackage path")
    ap.add_argument("--layer", default="buildings", help="Layer name in output GPKG")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    gdf = read_any(inp)
    if gdf is None or gdf.empty:
        raise SystemExit("Input produced empty GeoDataFrame")

    # ensure polygons only
    try:
        gdf = gdf[gdf.geometry.notna()]
        gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    except Exception:
        pass

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    else:
        gdf = gdf.to_crs("EPSG:4326")

    # Keep file size sane: drop huge unused columns if present
    drop_cols = [c for c in gdf.columns if c.lower() in ("wkt", "geom", "geometry_wkt")]
    if drop_cols:
        gdf = gdf.drop(columns=drop_cols)

    gdf.to_file(out, layer=args.layer, driver="GPKG")
    print(f"[OK] wrote {len(gdf)} footprints to {out} (layer={args.layer})")


if __name__ == "__main__":
    main()


