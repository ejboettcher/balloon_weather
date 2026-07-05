#!/usr/bin/env python3
"""
path_weather.py

Pull GFS weather (temp, RH, cloud cover, pressure) at each NOMADS GFS 0.25°
grid cell along a start→end path.  Each grid cell is evaluated at the time
the balloon arrives there, computed from its velocity.

Downloads one set of 2D fields (HGT, TMP, RH, TCDC) per unique GFS forecast
hour, then extracts values at every grid point for that hour — minimising
network calls.

Usage:
    uv run path_weather.py <start_lat> <start_lon> <start_alt>
                           <end_lat>   <end_lon>   <end_alt>
                           <velocity_ms>
                           [--time YYYY-MM-DDTHH:MM]

Example:
    uv run path_weather.py 39.74 -104.99 3000 40.10 -103.50 3200 15
    uv run path_weather.py 39.74 -104.99 3000 40.10 -103.50 3200 15 \\
        --time 2026-05-17T12:00
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
from herbie import Herbie

from gfs_grid_path import path_gfs_grid_points


def _first_var(ds) -> str:
    return list(ds.data_vars)[0]


def _xr(H: Herbie, search: str):
    """xarray() wrapper that always returns a single Dataset.

    Herbie returns a list when cfgrib cannot merge multiple GRIB messages
    (common with layer-integrated fields like TCDC).  We take the first
    element, which is the entire-atmosphere total for that pattern.
    """
    result = H.xarray(search, remove_grib=False)
    return result[0] if isinstance(result, list) else result


def _find_gfs_run(target_time: datetime) -> datetime:
    """Return the datetime of the most recent available GFS cycle ≤ target_time.

    Tries cycles from 5 h before target_time, stepping back in 6-hour
    increments. Timezone is preserved from target_time.
    """
    for hours_back in range(5, 30, 6):
        candidate  = target_time - timedelta(hours=hours_back)
        cycle_hour = (candidate.hour // 6) * 6
        run_dt     = candidate.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
        try:
            H = Herbie(
                run_dt.strftime("%Y-%m-%d %H:%M:%S"),
                model="gfs", product="pgrb2.0p25", fxx=0, verbose=False,
            )
            if H.grib:
                return run_dt
        except Exception:
            continue
    raise RuntimeError("No GFS run found — check your internet connection.")


def _herbie(run_dt: datetime, fxx: int) -> Herbie:
    return Herbie(
        run_dt.strftime("%Y-%m-%d %H:%M:%S"),
        model="gfs", product="pgrb2.0p25",
        fxx=fxx, verbose=False,
    )


def fetch_path_weather(pts: list[dict]) -> list[dict]:
    """Fetch GFS weather at each grid point using its arrival time.

    Parameters
    ----------
    pts : list[dict]
        Output of ``path_gfs_grid_points()``. Each entry has:
        lat, lon, alt, time, dist_km.

    Returns
    -------
    list[dict]
        One dict per grid cell, in path order, with keys:
        step, lat, lon, alt (m), time, dist_km, fxx,
        pressure_hpa, temp_c, rh_pct, cloud_cover_pct.
    """
    run_dt = _find_gfs_run(pts[0]["time"])
    print(f"GFS run : {run_dt.strftime('%Y-%m-%d %H:%M UTC')}")

    # Assign a GFS forecast hour to every grid point
    tagged = []
    for i, p in enumerate(pts):
        delta_h = (p["time"] - run_dt).total_seconds() / 3600
        fxx = int(round(max(0.0, delta_h)))
        if fxx > 120:               # GFS output becomes 3-hourly after fxx 120
            fxx = round(fxx / 3) * 3
        tagged.append({**p, "_idx": i, "_fxx": fxx})

    # Group by forecast hour so we download fields once per fxx
    by_fxx: dict[int, list[dict]] = defaultdict(list)
    for p in tagged:
        by_fxx[p["_fxx"]].append(p)

    results: list[dict | None] = [None] * len(pts)

    for fxx in sorted(by_fxx):
        group = by_fxx[fxx]
        n = len(group)
        print(f"\nfxx={fxx:03d}h  ({n} grid {'point' if n == 1 else 'points'})")
        H = _herbie(run_dt, fxx)

        print("  Downloading HGT, TMP, RH, TCDC ...")
        ds_hgt = _xr(H, r":HGT:\d+ mb:")
        ds_tmp = _xr(H, r":TMP:\d+ mb:")
        ds_rh  = _xr(H, r":RH:\d+ mb:")
        ds_cld = _xr(H, r":TCDC:entire atmosphere:")

        for p in group:
            lon_gfs = p["lon"] % 360

            # ── Altitude → nearest pressure level ────────────────────────────
            col_hgt  = ds_hgt.sel(latitude=p["lat"], longitude=lon_gfs, method="nearest")
            hgt_vals = col_hgt[_first_var(col_hgt)].values          # (n_levels,) m
            levs     = col_hgt["isobaricInhPa"].values               # hPa
            lev_idx  = int(np.argmin(np.abs(hgt_vals - p["alt"])))
            pres_hpa = float(levs[lev_idx])

            # ── Temperature ──────────────────────────────────────────────────
            col_tmp = ds_tmp.sel(latitude=p["lat"], longitude=lon_gfs, method="nearest")
            temp_k  = float(
                col_tmp[_first_var(col_tmp)]
                .sel(isobaricInhPa=pres_hpa, method="nearest").values
            )

            # ── Relative humidity ────────────────────────────────────────────
            col_rh = ds_rh.sel(latitude=p["lat"], longitude=lon_gfs, method="nearest")
            rh = float(
                col_rh[_first_var(col_rh)]
                .sel(isobaricInhPa=pres_hpa, method="nearest").values
            )

            # ── Total cloud cover (column-integrated, no pressure dim) ───────
            col_cld = ds_cld.sel(latitude=p["lat"], longitude=lon_gfs, method="nearest")
            cloud = float(col_cld[_first_var(col_cld)].values)

            results[p["_idx"]] = {
                "step"           : p["_idx"],
                "lat"            : p["lat"],
                "lon"            : p["lon"],
                "alt"            : p["alt"],
                "time"           : p["time"],
                "dist_km"        : p["dist_km"],
                "fxx"            : fxx,
                "pressure_hpa"   : pres_hpa,
                "temp_c"         : temp_k - 273.15,
                "rh_pct"         : rh,
                "cloud_cover_pct": cloud,
            }

    return results


def print_path_weather(results: list[dict]) -> None:
    sep = "─" * 98
    print(f"\n{sep}")
    print("  PATH WEATHER  (NOMADS GFS 0.25°)")
    print(sep)
    print(
        f"  {'#':>3}  {'Time':>5}  {'Lat':>8}  {'Lon(E)':>8}  "
        f"{'Alt m':>6}  {'hPa':>6}  {'T °C':>6}  "
        f"{'RH %':>5}  {'Cloud %':>7}  {'km':>6}  fxx"
    )
    print(sep)
    for r in results:
        t = r["time"]
        print(
            f"  {r['step']:>3}  {t.hour:02d}:{t.minute:02d}  "
            f"{r['lat']:>+8.3f}  {r['lon']:>8.3f}  "
            f"{r['alt']:>6.0f}  {r['pressure_hpa']:>6.1f}  "
            f"{r['temp_c']:>+6.1f}  {r['rh_pct']:>5.1f}  "
            f"{r['cloud_cover_pct']:>7.1f}  {r['dist_km']:>6.2f}  {r['fxx']}h"
        )
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch GFS weather at each 0.25° grid cell along a balloon path,\n"
            "using the arrival time at each cell computed from the balloon velocity."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run path_weather.py 39.74 -104.99 3000 40.10 -103.50 3200 15\n"
            "  uv run path_weather.py 39.74 -104.99 3000 40.10 -103.50 3200 15"
            " --time 2026-05-17T12:00\n"
        ),
    )
    parser.add_argument("start_lat", type=float, help="Start latitude  (°N)")
    parser.add_argument("start_lon", type=float, help="Start longitude (°E)")
    parser.add_argument("start_alt", type=float, help="Start altitude  (m ASL)")
    parser.add_argument("end_lat",   type=float, help="End latitude    (°N)")
    parser.add_argument("end_lon",   type=float, help="End longitude   (°E)")
    parser.add_argument("end_alt",   type=float, help="End altitude    (m ASL)")
    parser.add_argument("velocity",  type=float, help="Horizontal speed (m/s)")
    parser.add_argument(
        "--time", default=None, metavar="YYYY-MM-DDTHH:MM",
        help="Departure time UTC (default: now)",
    )
    args = parser.parse_args()

    start_time = (
        datetime.strptime(args.time, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
        if args.time
        else datetime.now(timezone.utc)
    )
    print(f"Departure : {start_time.strftime('%Y-%m-%d %H:%M UTC')}")

    pts = path_gfs_grid_points(
        start_lat=args.start_lat, start_lon=args.start_lon, start_alt=args.start_alt,
        end_lat=args.end_lat,     end_lon=args.end_lon,     end_alt=args.end_alt,
        start_time=start_time,    velocity_ms=args.velocity,
    )
    print(f"Path spans {len(pts)} GFS grid cell(s).\n")

    results = fetch_path_weather(pts)
    print_path_weather(results)


if __name__ == "__main__":
    main() # If I have the start and end points
