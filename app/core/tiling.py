"""AOI tiling utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .imaging import polygon_area_latlon_km2, sutherland_hodgman_clip, _equirect_xy


@dataclass
class TileCenter:
    id: str
    lat_deg: float
    lon_deg: float
    size_deg: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "lat_deg": self.lat_deg,
            "lon_deg": self.lon_deg,
            "size_deg": self.size_deg,
        }


def point_in_polygon(lat: float, lon: float, poly: List[Tuple[float, float]]) -> bool:
    """Ray-casting; poly is list of (lat, lon)."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        yi, xi = poly[i]
        yj, xj = poly[j]
        if ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-15) + xi
        ):
            inside = not inside
        j = i
    return inside


def adaptive_tile_size_km(off_nadir_deg: float, fov_deg: float, alt_km: float) -> float:
    """Approximate footprint size (along-look diagonal) at given off-nadir."""
    half = math.radians(fov_deg / 2.0)
    th = math.radians(off_nadir_deg)
    near = alt_km * math.tan(max(0.0, th - half))
    far = alt_km * math.tan(min(math.pi / 2 - 1e-3, th + half))
    along = max(2.0 * alt_km * math.tan(half), far - near)
    cross = 2.0 * alt_km * math.tan(half) / max(math.cos(th), 0.2)
    # Use the smaller dimension to get full coverage with overlap
    return min(along, cross) * 0.9   # 10% overlap factor


def km_to_deg_lat(km: float) -> float:
    return km / 111.0


def km_to_deg_lon(km: float, lat_deg: float) -> float:
    return km / (111.0 * max(math.cos(math.radians(lat_deg)), 0.2))


def tile_aoi(
    aoi_polygon: List[Tuple[float, float]],
    tile_size_km: float,
) -> List[TileCenter]:
    """Build a regular lat/lon grid covering the AOI bbox; keep tiles whose
    center is inside the polygon.
    """
    lats = [p[0] for p in aoi_polygon]
    lons = [p[1] for p in aoi_polygon]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    lat_c = 0.5 * (lat_min + lat_max)

    d_lat = km_to_deg_lat(tile_size_km)
    d_lon = km_to_deg_lon(tile_size_km, lat_c)

    n_lat = max(1, int(math.ceil((lat_max - lat_min) / d_lat)))
    n_lon = max(1, int(math.ceil((lon_max - lon_min) / d_lon)))

    # Center the grid in the bbox
    used_lat = n_lat * d_lat
    used_lon = n_lon * d_lon
    lat0 = lat_min + (lat_max - lat_min - used_lat) / 2.0 + d_lat / 2.0
    lon0 = lon_min + (lon_max - lon_min - used_lon) / 2.0 + d_lon / 2.0

    tiles: List[TileCenter] = []
    for i in range(n_lat):
        for j in range(n_lon):
            lat = lat0 + i * d_lat
            lon = lon0 + j * d_lon
            if point_in_polygon(lat, lon, aoi_polygon):
                tiles.append(
                    TileCenter(
                        id=f"t_{i:02d}_{j:02d}",
                        lat_deg=lat,
                        lon_deg=lon,
                        size_deg=max(d_lat, d_lon),
                    )
                )
    return tiles


def tile_polygon(tile: TileCenter) -> List[Tuple[float, float]]:
    """Return tile rectangle as 4 (lat, lon) corners in CCW order."""
    half = tile.size_deg / 2.0
    return [
        (tile.lat_deg - half, tile.lon_deg - half),
        (tile.lat_deg - half, tile.lon_deg + half),
        (tile.lat_deg + half, tile.lon_deg + half),
        (tile.lat_deg + half, tile.lon_deg - half),
    ]


def check_tile_coverage(
    tile_poly: List[Tuple[float, float]],
    footprint_poly: List[Tuple[float, float]],
) -> float:
    """Fraction of `tile_poly` covered by `footprint_poly`."""
    if len(tile_poly) < 3 or len(footprint_poly) < 3:
        return 0.0
    lat0 = sum(p[0] for p in tile_poly) / len(tile_poly)
    tile_xy = [tuple(row) for row in _equirect_xy(tile_poly, lat0)]
    foot_xy = [tuple(row) for row in _equirect_xy(footprint_poly, lat0)]
    # Ensure clip polygon is CCW for Sutherland-Hodgman
    if _signed_area(foot_xy) < 0:
        foot_xy = list(reversed(foot_xy))
    if _signed_area(tile_xy) < 0:
        tile_xy = list(reversed(tile_xy))
    clipped = sutherland_hodgman_clip(tile_xy, foot_xy)
    if not clipped:
        return 0.0
    a_clipped = abs(_signed_area(clipped))
    a_tile = abs(_signed_area(tile_xy))
    if a_tile <= 0:
        return 0.0
    return min(1.0, a_clipped / a_tile)


def _signed_area(poly) -> float:
    if len(poly) < 3:
        return 0.0
    arr = np.asarray(poly, dtype=float)
    x = arr[:, 0]
    y = arr[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
