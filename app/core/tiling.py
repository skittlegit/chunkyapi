"""AOI tiling and polygon utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple


LatLon = Tuple[float, float]  # (lat_deg, lon_deg)


@dataclass
class TileCenter:
    id: str
    lat_deg: float
    lon_deg: float
    row: int
    col: int


def polygon_bbox(poly: Sequence[LatLon]) -> Tuple[float, float, float, float]:
    lats = [p[0] for p in poly]
    lons = [p[1] for p in poly]
    return min(lats), min(lons), max(lats), max(lons)


def point_in_polygon(point: LatLon, poly: Sequence[LatLon]) -> bool:
    """Standard ray-casting in (lat, lon) plane. Adequate for small AOIs."""
    lat, lon = point
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        lat_i, lon_i = poly[i]
        lat_j, lon_j = poly[j]
        if (lon_i > lon) != (lon_j > lon):
            x_intersect = (lat_j - lat_i) * (lon - lon_i) / ((lon_j - lon_i) or 1e-12) + lat_i
            if lat < x_intersect:
                inside = not inside
        j = i
    return inside


def adaptive_tile_size_km(off_nadir_deg: float, fov_deg: float, altitude_km: float) -> float:
    """Approx tile spacing (km) based on expected ground footprint, with overlap."""
    half = math.radians(fov_deg / 2.0)
    theta = math.radians(off_nadir_deg)
    # Along-look stretch
    along = altitude_km * (math.tan(theta + half) - math.tan(theta - half))
    # Cross-look ~ 2*h*tan(half) / cos(theta)
    cross = 2.0 * altitude_km * math.tan(half) / max(math.cos(theta), 0.05)
    tile_km = min(along, cross) * 0.85  # 15% overlap
    return max(5.0, tile_km)


def km_to_deg(lat_deg: float) -> Tuple[float, float]:
    KM_PER_DEG_LAT = 110.574
    KM_PER_DEG_LON = 111.320 * max(math.cos(math.radians(lat_deg)), 1e-3)
    return KM_PER_DEG_LAT, KM_PER_DEG_LON


def tile_aoi(
    aoi_polygon: Sequence[LatLon], tile_size_km: float
) -> List[TileCenter]:
    lat_min, lon_min, lat_max, lon_max = polygon_bbox(aoi_polygon)
    lat_c = 0.5 * (lat_min + lat_max)
    km_lat, km_lon = km_to_deg(lat_c)
    d_lat = tile_size_km / km_lat
    d_lon = tile_size_km / km_lon

    n_lat = max(1, int(math.ceil((lat_max - lat_min) / d_lat)))
    n_lon = max(1, int(math.ceil((lon_max - lon_min) / d_lon)))

    tiles: List[TileCenter] = []
    for i in range(n_lat):
        lat = lat_min + (i + 0.5) * d_lat
        for j in range(n_lon):
            lon = lon_min + (j + 0.5) * d_lon
            if point_in_polygon((lat, lon), aoi_polygon):
                tiles.append(
                    TileCenter(
                        id=f"t_{i}_{j}",
                        lat_deg=lat,
                        lon_deg=lon,
                        row=i,
                        col=j,
                    )
                )
    return tiles


def boustrophedon_order(tiles: List[TileCenter]) -> List[TileCenter]:
    """Sort tiles row-major with alternating row directions."""
    by_row: dict[int, list[TileCenter]] = {}
    for t in tiles:
        by_row.setdefault(t.row, []).append(t)
    out: List[TileCenter] = []
    for idx, row in enumerate(sorted(by_row.keys())):
        row_tiles = sorted(by_row[row], key=lambda t: t.col)
        if idx % 2 == 1:
            row_tiles.reverse()
        out.extend(row_tiles)
    return out


def polygon_area_km2(poly: Sequence[LatLon]) -> float:
    if len(poly) < 3:
        return 0.0
    lat0 = sum(p[0] for p in poly) / len(poly)
    KM_PER_DEG_LAT = 110.574
    KM_PER_DEG_LON = 111.320 * math.cos(math.radians(lat0))
    xy = [((lon - poly[0][1]) * KM_PER_DEG_LON, (lat - poly[0][0]) * KM_PER_DEG_LAT) for (lat, lon) in poly]
    s = 0.0
    n = len(xy)
    for i in range(n):
        x1, y1 = xy[i]
        x2, y2 = xy[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5
