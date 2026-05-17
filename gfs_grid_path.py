#!/usr/bin/env python3
"""
gfs_grid_path.py

Given a start/end position, altitude, time, and speed, return the
lat/lon/alt/time at which a traveller enters each NOMADS GFS 0.25° grid cell
along the path.

Useful for pre-building the exact list of grid points to query before making
GFS/Herbie network calls along a balloon trajectory.

Example
-------
    from datetime import datetime, timezone
    from gfs_grid_path import path_gfs_grid_points

    pts = path_gfs_grid_points(
        start_lat=39.74, start_lon=-104.99, start_alt=3000,
        end_lat=40.10,   end_lon=-103.50,   end_alt=3200,
        start_time=datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc),
        velocity_ms=15.0,
    )
    for p in pts:
        print(p)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

EARTH_RADIUS_M  = 6_371_000
GFS_RESOLUTION  = 0.25          # degrees (default GFS 0.25° product)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a  = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _slerp(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    frac: float,
) -> tuple[float, float]:
    """Spherical linear interpolation (great-circle) at fraction frac ∈ [0, 1]."""
    def to_ecef(lat_deg, lon_deg):
        φ, λ = math.radians(lat_deg), math.radians(lon_deg)
        return math.cos(φ) * math.cos(λ), math.cos(φ) * math.sin(λ), math.sin(φ)

    x1, y1, z1 = to_ecef(lat1, lon1)
    x2, y2, z2 = to_ecef(lat2, lon2)

    dot   = max(-1.0, min(1.0, x1 * x2 + y1 * y2 + z1 * z2))
    omega = math.acos(dot)

    if omega < 1e-10:               # coincident points
        return lat1, lon1

    s  = math.sin(omega)
    s1 = math.sin((1.0 - frac) * omega) / s
    s2 = math.sin(frac * omega) / s

    x, y, z = s1 * x1 + s2 * x2, s1 * y1 + s2 * y2, s1 * z1 + s2 * z2
    return math.degrees(math.asin(z)), math.degrees(math.atan2(y, x))


def _snap(value: float, resolution: float) -> float:
    """Round to the nearest grid point at the given resolution."""
    return round(value / resolution) * resolution


# ── Public API ────────────────────────────────────────────────────────────────

def path_gfs_grid_points(
    start_lat: float,
    start_lon: float,
    start_alt: float,
    end_lat: float,
    end_lon: float,
    end_alt: float,
    start_time: datetime,
    velocity_ms: float,
    gfs_res: float = GFS_RESOLUTION,
) -> list[dict]:
    """Return one entry per GFS grid cell the path passes through.

    The path is treated as a great-circle arc at constant speed. Altitude
    is linearly interpolated between start_alt and end_alt. The function
    samples the arc at a spacing of ~1/5 of a grid cell length, so no
    crossing is missed.

    Parameters
    ----------
    start_lat, start_lon : float
        Departure position (°N / °E; west is negative).
    start_alt : float
        Departure altitude (m ASL).
    end_lat, end_lon : float
        Destination position (°N / °E).
    end_alt : float
        Destination altitude (m ASL).
    start_time : datetime
        Departure time. Timezone-aware preferred.
    velocity_ms : float
        Constant horizontal speed along the great-circle path (m/s).
    gfs_res : float
        GFS grid resolution in degrees (default 0.25 for the 0.25° product).

    Returns
    -------
    list[dict]  – one entry per unique GFS grid cell entered, in order:
        lat      (float)    – GFS grid latitude  (°N, snapped to gfs_res)
        lon      (float)    – GFS grid longitude (°E, 0–360, snapped to gfs_res)
        alt      (float)    – interpolated altitude at grid-cell entry (m ASL)
        time     (datetime) – clock time at grid-cell entry
        dist_km  (float)    – cumulative great-circle distance from start (km)
    """
    if velocity_ms <= 0:
        raise ValueError(f"velocity_ms must be positive, got {velocity_ms}")

    total_dist_m = _haversine_m(start_lat, start_lon, end_lat, end_lon)

    # Same grid cell — return immediately
    if total_dist_m < 1.0:
        return [{
            "lat"    : _snap(start_lat, gfs_res),
            "lon"    : _snap(start_lon % 360, gfs_res) % 360,
            "alt"    : start_alt,
            "time"   : start_time,
            "dist_km": 0.0,
        }]

    total_time_s = total_dist_m / velocity_ms

    # Step size ≈ 1/5 of one grid-cell length — catches every crossing
    cell_m  = gfs_res * (math.pi / 180.0) * EARTH_RADIUS_M   # ~27.8 km at equator
    step_m  = cell_m / 5.0
    n_steps = max(int(total_dist_m / step_m) + 1, 10)

    results  = []
    prev_key = None

    for i in range(n_steps + 1):
        frac = i / n_steps

        lat, lon = _slerp(start_lat, start_lon, end_lat, end_lon, frac)
        alt      = start_alt + (end_alt - start_alt) * frac
        t        = start_time + timedelta(seconds=total_time_s * frac)
        dist_km  = _haversine_m(start_lat, start_lon, lat, lon) / 1000.0

        grid_lat = max(-90.0, min(90.0, _snap(lat, gfs_res)))
        grid_lon = _snap(lon % 360, gfs_res) % 360   # normalise to 0–360

        key = (grid_lat, grid_lon)
        if key != prev_key:
            results.append({
                "lat"    : grid_lat,
                "lon"    : grid_lon,
                "alt"    : alt,
                "time"   : t,
                "dist_km": dist_km,
            })
            prev_key = key

    return results


# ── CLI / quick test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from datetime import timezone

    parser = argparse.ArgumentParser(
        description="List GFS 0.25° grid cells along a straight-line path.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python gfs_grid_path.py "
            "39.74 -104.99 3000 40.10 -103.50 3200 15\n"
        ),
    )
    parser.add_argument("start_lat",  type=float, help="Start latitude  (°N)")
    parser.add_argument("start_lon",  type=float, help="Start longitude (°E)")
    parser.add_argument("start_alt",  type=float, help="Start altitude  (m ASL)")
    parser.add_argument("end_lat",    type=float, help="End latitude    (°N)")
    parser.add_argument("end_lon",    type=float, help="End longitude   (°E)")
    parser.add_argument("end_alt",    type=float, help="End altitude    (m ASL)")
    parser.add_argument("velocity",   type=float, help="Speed           (m/s)")
    parser.add_argument("--res",      type=float, default=GFS_RESOLUTION,
                        help="GFS grid resolution in degrees (default 0.25)")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    pts = path_gfs_grid_points(
        start_lat=args.start_lat, start_lon=args.start_lon, start_alt=args.start_alt,
        end_lat=args.end_lat,     end_lon=args.end_lon,     end_alt=args.end_alt,
        start_time=now, velocity_ms=args.velocity, gfs_res=args.res,
    )

    sep = "─" * 68
    print(f"\n{sep}")
    print(f"  GFS {args.res}° grid cells along path  ({len(pts)} cells)")
    print(sep)
    print(f"  {'#':>3}  {'Lat':>8}  {'Lon (0-360)':>11}  "
          f"{'Alt (m)':>8}  {'Dist (km)':>9}  Time (UTC)")
    print(sep)
    for i, p in enumerate(pts):
        print(
            f"  {i:>3}  {p['lat']:>+8.3f}  {p['lon']:>11.3f}  "
            f"{p['alt']:>8.0f}  {p['dist_km']:>9.2f}  "
            f"{p['time'].strftime('%H:%M:%S')}"
        )
    print(sep)
