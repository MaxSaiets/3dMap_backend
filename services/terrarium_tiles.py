"""
Terrarium elevation tiles loader (no API key).

Tile source (commonly used Mapzen terrarium):
https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png

Terrarium encoding:
elevation_m = (R * 256 + G + B / 256) - 32768

This module downloads required tiles on demand, caches them on disk, and provides
fast sampling for arrays of (lat, lon) points.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import requests


@dataclass(frozen=True)
class TileKey:
    z: int
    x: int
    y: int


def _latlon_to_global_pixel(lon: float, lat: float, z: int) -> Tuple[float, float]:
    # WebMercator global pixel coordinates at zoom z
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 256.0 * (2**z)
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n
    return x, y


def _global_pixel_to_tile(x: float, y: float) -> Tuple[int, int, float, float]:
    tx = int(math.floor(x / 256.0))
    ty = int(math.floor(y / 256.0))
    px = x - tx * 256.0
    py = y - ty * 256.0
    return tx, ty, px, py


def _bilinear_sample(img: np.ndarray, px: float, py: float) -> float:
    # img: (H,W) float32
    h, w = img.shape
    x = float(np.clip(px, 0.0, w - 1.0))
    y = float(np.clip(py, 0.0, h - 1.0))
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx = x - x0
    dy = y - y0
    v00 = img[y0, x0]
    v10 = img[y0, x1]
    v01 = img[y1, x0]
    v11 = img[y1, x1]
    return float((v00 * (1 - dx) + v10 * dx) * (1 - dy) + (v01 * (1 - dx) + v11 * dx) * dy)


def _decode_terrarium_png(png_bytes: bytes) -> np.ndarray:
    # Decode PNG bytes -> elevation array float32 (256x256)
    from PIL import Image
    import io

    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.asarray(im, dtype=np.float32)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    elev = (r * 256.0 + g + b / 256.0) - 32768.0
    return elev.astype(np.float32)


class TerrariumTileProvider:
    def __init__(
        self,
        base_url: str = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium",
        cache_dir: str = "cache/terrarium",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

        # in-memory decoded tiles cache
        self._mem: Dict[TileKey, np.ndarray] = {}

    def _tile_path(self, key: TileKey) -> Path:
        return self.cache_dir / str(key.z) / str(key.x) / f"{key.y}.png"

    def _fetch_tile_png(self, key: TileKey) -> Optional[bytes]:
        # disk cache first
        p = self._tile_path(key)
        if p.exists():
            return p.read_bytes()

        url = f"{self.base_url}/{key.z}/{key.x}/{key.y}.png"
        try:
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                return None
            png = resp.content
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(png)
            return png
        except Exception:
            return None

    def get_tile(self, key: TileKey) -> Optional[np.ndarray]:
        if key in self._mem:
            return self._mem[key]
        png = self._fetch_tile_png(key)
        if png is None:
            return None
        elev = _decode_terrarium_png(png)
        self._mem[key] = elev
        return elev

    def sample_points(self, lats: np.ndarray, lons: np.ndarray, z: int) -> Optional[np.ndarray]:
        if lats.size == 0:
            return np.array([])

        out = np.empty_like(lats, dtype=np.float32)
        out.fill(np.nan)

        # group points by tile for efficiency
        by_tile: Dict[TileKey, list[tuple[int, float, float]]] = {}
        for i in range(lats.size):
            gx, gy = _latlon_to_global_pixel(float(lons[i]), float(lats[i]), z)
            tx, ty, px, py = _global_pixel_to_tile(gx, gy)
            key = TileKey(z=z, x=tx, y=ty)
            by_tile.setdefault(key, []).append((i, px, py))

        for key, pts in by_tile.items():
            tile = self.get_tile(key)
            if tile is None:
                continue
            for idx, px, py in pts:
                out[idx] = _bilinear_sample(tile, px, py)

        return out


