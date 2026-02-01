"""
Download Microsoft GlobalMLBuildingFootprints for Ukraine and build a bbox-queryable GeoPackage.

Source manifest:
  https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv

The referenced files end with .csv.gz but the contents are GeoJSONL (one GeoJSON Feature per line).

Outputs:
  - cache/footprints/ms/dataset-links.csv
  - cache/footprints/ms/ukraine/*.csv.gz   (raw partitions)
  - cache/footprints/ukraine_buildings.gpkg (layer: buildings, EPSG:4326)

Run:
  python scripts/download_ms_footprints_ukraine.py
  python scripts/download_ms_footprints_ukraine.py --no-gpkg   (download only)
  python scripts/download_ms_footprints_ukraine.py --overwrite-gpkg
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Set

import pandas as pd
import requests
import geopandas as gpd
from shapely.geometry import shape


MANIFEST_URL = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"


def _parse_size_to_bytes(s: str) -> int:
    s = str(s).strip()
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*([KMG]?B)$", s, flags=re.I)
    if not m:
        return 0
    v = float(m.group(1))
    u = m.group(2).upper()
    mult = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}.get(u, 1)
    return int(v * mult)


def _download_file(url: str, dst: Path, expected_bytes: Optional[int] = None, timeout: int = 120) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    # If a partial file exists from a previous run, delete it so we can resume cleanly.
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    if dst.exists():
        # If we know expected size and it doesn't match, re-download.
        if expected_bytes and dst.stat().st_size != expected_bytes:
            try:
                dst.unlink()
            except Exception:
                pass
        else:
            return

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(dst)


def _load_manifest(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        data = requests.get(MANIFEST_URL, timeout=120).content
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
    df = pd.read_csv(cache_path)
    return df


def _iter_geojsonl_features_gz(path: Path) -> Iterator[Dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _extract_row(feature: Dict) -> Optional[Dict]:
    """
    Converts GeoJSON Feature -> row dict with 'geometry' as shapely geometry.
    Keeps only a compact set of properties for size/perf.
    """
    if not feature or feature.get("type") != "Feature":
        return None

    geom = feature.get("geometry")
    if not geom:
        return None
    try:
        g = shape(geom)
    except Exception:
        return None
    if g.is_empty:
        return None

    props = feature.get("properties") or {}
    # Known common fields across versions
    out: Dict = {
        "geometry": g,
    }
    for k in [
        "height",
        "confidence",
        "source",
        "area",
        "class",
        "building",
        "roof:shape",
        "roof:height",
    ]:
        if k in props:
            out[k] = props.get(k)
    # Preserve any id if present
    for k in ["id", "fid", "Id", "ID"]:
        if k in props:
            out["footprint_id"] = props.get(k)
            break
    return out


def _progress_path_for(output_gpkg: Path) -> Path:
    return output_gpkg.with_suffix(output_gpkg.suffix + ".progress.json")


def _load_progress(progress_path: Path) -> Set[str]:
    if not progress_path.exists():
        return set()
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
        files = payload.get("processed_files") or []
        return set(str(x) for x in files)
    except Exception:
        return set()


def _save_progress(progress_path: Path, processed_files: Set[str]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"processed_files": sorted(processed_files)}
    progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_gpkg_from_partitions(
    partitions_dir: Path,
    output_gpkg: Path,
    layer: str = "buildings",
    overwrite: bool = False,
    chunk_size: int = 5000,
) -> None:
    progress_path = _progress_path_for(output_gpkg)

    if overwrite:
        if output_gpkg.exists():
            output_gpkg.unlink()
        if progress_path.exists():
            try:
                progress_path.unlink()
            except Exception:
                pass
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(partitions_dir.glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No partitions found in: {partitions_dir}")

    # Safety: if the gpkg exists but we don't have progress, we can't safely resume without risking duplicates.
    if output_gpkg.exists() and not progress_path.exists():
        raise RuntimeError(
            f"GPKG already exists but no progress checkpoint found: {output_gpkg}\n"
            f"To rebuild safely, rerun with --overwrite-gpkg."
        )

    processed = _load_progress(progress_path)
    first_write = not output_gpkg.exists()

    total_written = 0
    for i, gz_path in enumerate(files, start=1):
        if gz_path.name in processed:
            # already processed in a previous run
            continue
        batch: List[Dict] = []
        for feat in _iter_geojsonl_features_gz(gz_path):
            row = _extract_row(feat)
            if row is None:
                continue
            batch.append(row)
            if len(batch) >= chunk_size:
                gdf = gpd.GeoDataFrame(batch, geometry="geometry", crs="EPSG:4326")
                gdf.to_file(
                    output_gpkg,
                    layer=layer,
                    driver="GPKG",
                    engine="fiona",
                    mode="w" if first_write else "a",
                    index=False,
                )
                total_written += len(gdf)
                batch.clear()
                first_write = False

        if batch:
            gdf = gpd.GeoDataFrame(batch, geometry="geometry", crs="EPSG:4326")
            gdf.to_file(
                output_gpkg,
                layer=layer,
                driver="GPKG",
                engine="fiona",
                mode="w" if first_write else "a",
                index=False,
            )
            total_written += len(gdf)
            first_write = False

        processed.add(gz_path.name)
        _save_progress(progress_path, processed)
        print(f"[gpkg] {len(processed)}/{len(files)} processed: {gz_path.name} (total_rows={total_written})")

    # Finished successfully -> remove checkpoint
    try:
        progress_path.unlink()
    except Exception:
        pass
    print(f"[OK] GPKG ready: {output_gpkg} (rows={total_written}, layer={layer})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="cache/footprints/ms/ukraine", help="Where to store raw Ukraine partitions")
    ap.add_argument("--manifest", default="cache/footprints/ms/dataset-links.csv", help="Manifest cache path")
    ap.add_argument("--no-gpkg", action="store_true", help="Download only; don't build the GeoPackage")
    ap.add_argument("--gpkg", default="cache/footprints/ukraine_buildings.gpkg", help="Output GeoPackage path")
    ap.add_argument("--layer", default="buildings", help="Output GeoPackage layer name")
    ap.add_argument("--overwrite-gpkg", action="store_true", help="Overwrite existing gpkg output")
    ap.add_argument("--max-files", type=int, default=0, help="Limit number of partitions to download (debug)")
    ap.add_argument("--chunk-size", type=int, default=15000, help="Rows per write batch (bigger = faster, more RAM)")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    gpkg_path = Path(args.gpkg)

    df = _load_manifest(manifest_path)
    uk = df[df["Location"].astype(str).str.lower().eq("ukraine")].copy()
    if uk.empty:
        raise SystemExit("No Ukraine rows found in manifest")

    uk["size_bytes"] = uk["Size"].map(_parse_size_to_bytes)
    total_gib = uk["size_bytes"].sum() / (1024**3)
    print(f"[ms] Ukraine partitions: {len(uk)} files, ~{total_gib:.2f} GiB compressed")

    rows = uk.to_dict(orient="records")
    if args.max_files and args.max_files > 0:
        rows = rows[: args.max_files]

    # Download
    for i, row in enumerate(rows, start=1):
        url = row["Url"]
        quadkey = str(row.get("QuadKey") or "")
        # Keep original filename but prefix quadkey for stable naming
        name = url.split("/")[-1]
        dst = out_dir / f"{quadkey}_{name}"
        expected = int(row.get("size_bytes") or 0) or None
        _download_file(url, dst, expected_bytes=expected, timeout=180)
        if i % 10 == 0 or i == len(rows):
            print(f"[dl] {i}/{len(rows)} downloaded")

    print(f"[OK] Downloaded to: {out_dir}")

    if not args.no_gpkg:
        build_gpkg_from_partitions(
            partitions_dir=out_dir,
            output_gpkg=gpkg_path,
            layer=args.layer,
            overwrite=args.overwrite_gpkg,
            chunk_size=int(args.chunk_size),
        )


if __name__ == "__main__":
    main()


