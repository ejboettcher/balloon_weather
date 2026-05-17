#!/usr/bin/env python3
"""
balloon_weather.py

Fetch NOAA GFS weather via Herbie and predict balloon drift over N 30-minute steps.

Usage:
    python balloon_weather.py <lat> <lon> <altitude_m> [n_steps]

Examples:
    python balloon_weather.py 39.7392 -104.9903 3000        # single 30-min step
    python balloon_weather.py 39.7392 -104.9903 3000 6      # 3 hours (6 × 30 min)

Dependencies:
    pip install herbie-data numpy matplotlib cartopy
    (herbie-data pulls in xarray, cfgrib, eccodes, pandas automatically)
"""

import argparse
import math
from datetime import datetime, timedelta, timezone

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from herbie import Herbie

EARTH_RADIUS_M = 6_371_000


def find_latest_gfs() -> Herbie:
    """Return a Herbie handle for the most recent available GFS 0.25° analysis.

    GFS runs 00/06/12/18 UTC and is typically available ~4 h after cycle time.
    Tries cycles starting 5 h back and steps in 6-hour increments.
    """
    now = datetime.now(timezone.utc)
    for hours_back in range(5, 30, 6):
        candidate  = now - timedelta(hours=hours_back)
        cycle_hour = (candidate.hour // 6) * 6
        run_dt     = candidate.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
        run_dt_str = run_dt.strftime("%Y-%m-%d %H:%M:%S")
        print(run_dt_str)
        try:
            H = Herbie(run_dt_str, model="gfs", product="pgrb2.0p25", fxx=0, verbose=False)
            if H.grib:
                return H
        except Exception:
            continue
    raise RuntimeError("No GFS data found — check your internet connection.")


def _first_var(ds) -> str:
    """Name of the first non-coordinate data variable in a Dataset."""
    return list(ds.data_vars)[0]


def fetch_weather(lat: float, lon: float, altitude_m: float) -> tuple[Herbie, dict]:
    """Download GFS fields at the start point.

    Returns (Herbie handle, weather dict). Keeping the handle lets the caller
    reuse the same GFS run for the trajectory download without re-searching.
    """
    print("Searching for latest GFS run...")
    H = find_latest_gfs()
    print(f"GFS run : {H.date}  (fxx={H.fxx:03d})")

    lon_gfs = lon % 360  # GFS uses 0–360° longitude

    # ── Step 1: geopotential height column → map altitude to pressure level ──
    print("Fetching geopotential height column...")
    ds_hgt  = H.xarray(r":HGT:\d+ mb:", remove_grib=False)
    col_hgt = ds_hgt.sel(latitude=lat, longitude=lon_gfs, method="nearest")

    hgt_vals     = col_hgt[_first_var(col_hgt)].values    # (n_levels,) in metres
    levs         = col_hgt["isobaricInhPa"].values         # hPa
    lev_idx      = int(np.argmin(np.abs(hgt_vals - altitude_m)))
    pressure_hpa = float(levs[lev_idx])
    model_alt_m  = float(hgt_vals[lev_idx])
    nearest_lat  = float(col_hgt.latitude.values)
    nearest_lon  = float(col_hgt.longitude.values)

    print(f"Grid pt : {nearest_lat:.3f}°N, {nearest_lon:.3f}°E")
    print(f"Level   : {altitude_m:.0f} m → {pressure_hpa:.1f} hPa  (model {model_alt_m:.0f} m)")

    # ── Step 2: point values at the identified pressure level ─────────────────
    def at_level(search: str) -> float:
        ds  = H.xarray(search, remove_grib=False)
        col = ds.sel(latitude=lat, longitude=lon_gfs, method="nearest")
        return float(
            col[_first_var(col)].sel(isobaricInhPa=pressure_hpa, method="nearest").values
        )

    print("Fetching temperature, RH, and winds...")
    temp_k = at_level(r":TMP:\d+ mb:")
    rh     = at_level(r":RH:\d+ mb:")
    u_wind = at_level(r":UGRD:\d+ mb:")   # eastward  m/s
    v_wind = at_level(r":VGRD:\d+ mb:")   # northward m/s

    print("Fetching cloud cover...")
    ds_cld = H.xarray(r":TCDC:entire atmosphere:", remove_grib=False)
    cld_pt = ds_cld.sel(latitude=lat, longitude=lon_gfs, method="nearest")
    cloud  = float(cld_pt[_first_var(cld_pt)].values)

    wind_speed = math.hypot(u_wind, v_wind)
    wind_dir   = (270 - math.degrees(math.atan2(v_wind, u_wind))) % 360

    wx = {
        "nearest_lat"    : nearest_lat,
        "nearest_lon"    : nearest_lon,
        "altitude_m"     : altitude_m,
        "model_alt_m"    : model_alt_m,
        "pressure_hpa"   : pressure_hpa,
        "temp_c"         : temp_k - 273.15,
        "rh_pct"         : rh,
        "cloud_cover_pct": cloud,
        "u_wind_ms"      : u_wind,
        "v_wind_ms"      : v_wind,
        "wind_speed_ms"  : wind_speed,
        "wind_dir_deg"   : wind_dir,
    }
    return H, wx


def compute_trajectory(
    H: Herbie, start_lat: float, start_lon: float, wx: dict, n_steps: int
) -> list[dict]:
    """Compute balloon position for n_steps × 30-minute intervals.

    Downloads the full 2D U/V wind fields at the identified pressure level once,
    then does spatial lookups at each new position — no repeated network calls.
    Wind is sampled at the current position before each advance, so the path
    bends naturally as the balloon crosses different wind regimes.
    """
    pressure_hpa = wx["pressure_hpa"]
    pres_label   = int(pressure_hpa)   # e.g. 700 → ":UGRD:700 mb:"

    print(f"\nDownloading 2D wind fields at {pres_label} hPa for trajectory...")
    ds_u    = H.xarray(f":UGRD:{pres_label} mb:", remove_grib=False)
    ds_v    = H.xarray(f":VGRD:{pres_label} mb:", remove_grib=False)
    u_field = ds_u[_first_var(ds_u)]   # DataArray (latitude × longitude)
    v_field = ds_v[_first_var(ds_v)]

    trajectory = [{"step": 0, "time_min": 0,
                   "lat": start_lat, "lon": start_lon, "dist_km": 0.0}]
    curr_lat = start_lat
    curr_lon = start_lon

    for i in range(1, n_steps + 1):
        lon_gfs = curr_lon % 360
        u = float(u_field.sel(latitude=curr_lat, longitude=lon_gfs, method="nearest").values)
        v = float(v_field.sel(latitude=curr_lat, longitude=lon_gfs, method="nearest").values)

        dt_sec   = 30 * 60
        lat_rad  = math.radians(curr_lat)
        curr_lat += math.degrees(v * dt_sec / EARTH_RADIUS_M)
        curr_lon += math.degrees(u * dt_sec / (EARTH_RADIUS_M * math.cos(lat_rad)))

        trajectory.append({
            "step"    : i,
            "time_min": i * 30,
            "lat"     : curr_lat,
            "lon"     : curr_lon,
            "dist_km" : haversine_km(start_lat, start_lon, curr_lat, curr_lon),
        })

    return trajectory


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R  = EARTH_RADIUS_M / 1000
    φ1 = math.radians(lat1);  φ2 = math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a  = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def print_weather(lat: float, lon: float, wx: dict) -> None:
    sep = "─" * 54
    print(f"\n{sep}")
    print("  BALLOON WEATHER REPORT  (NOMADS GFS 0.25°)")
    print(sep)
    print(f"  Query : {lat:+.4f}°  {lon:+.4f}°  {wx['altitude_m']:.0f} m ASL")
    print(f"  Model : {wx['nearest_lat']:+.4f}°  {wx['nearest_lon']:+.4f}°  "
          f"{wx['model_alt_m']:.0f} m  ({wx['pressure_hpa']:.1f} hPa)")
    print(sep)
    print(f"  Temperature   : {wx['temp_c']:+.1f} °C")
    print(f"  Rel. Humidity : {wx['rh_pct']:.1f} %")
    print(f"  Cloud Cover   : {wx['cloud_cover_pct']:.1f} %  (total column)")
    print(f"  Wind (U, E+)  : {wx['u_wind_ms']:+.2f} m/s")
    print(f"  Wind (V, N+)  : {wx['v_wind_ms']:+.2f} m/s")
    print(f"  Wind speed    : {wx['wind_speed_ms']:.2f} m/s  "
          f"({wx['wind_speed_ms'] * 1.94384:.1f} kt)")
    print(f"  Wind from     : {wx['wind_dir_deg']:.0f}°")
    print(sep)


def print_trajectory(trajectory: list[dict], altitude_m: float, pressure_hpa: float) -> None:
    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  BALLOON TRAJECTORY  ({altitude_m:.0f} m ASL / {pressure_hpa:.1f} hPa)")
    print(sep)
    print(f"  {'Step':>4}  {'HH:MM':>5}  {'Latitude':>11}  {'Longitude':>12}  {'Dist (km)':>9}")
    print(sep)
    for s in trajectory:
        h   = s["time_min"] // 60
        m   = s["time_min"] % 60
        tag = "  ← start" if s["step"] == 0 else ""
        print(
            f"  {s['step']:>4}  {h:02d}:{m:02d}  "
            f"{s['lat']:>+11.5f}  {s['lon']:>+12.5f}  "
            f"{s['dist_km']:>9.2f}{tag}"
        )
    print(sep)
    final = trajectory[-1]
    total_min = final["time_min"]
    h, m = total_min // 60, total_min % 60
    print(f"  Total drift: {final['dist_km']:.2f} km  "
          f"({final['dist_km'] * 0.539957:.1f} nm)  "
          f"over {h:02d}:{m:02d} (hh:mm)")
    print(sep)


def plot_trajectory(
    H: Herbie,
    trajectory: list[dict],
    wx: dict,
    output_path: str = "balloon_trajectory.png",
) -> plt.Figure:
    """Plot trajectory on a GFS cloud cover heatmap using a Geostationary projection.

    The map is centered on the trajectory midpoint and zoomed so the full path
    plus a margin fills the frame.
    """
    # ── Cloud cover 2D field (Herbie cache: already downloaded, no new fetch) ──
    print("Loading 2D cloud cover field for map...")
    ds_cld = H.xarray(r":TCDC:entire atmosphere:", remove_grib=False)
    tcc    = ds_cld[_first_var(ds_cld)]          # (latitude, longitude)
    lats_g = tcc.latitude.values                  # ascending
    lons_g = tcc.longitude.values                 # 0–360
    data_g = tcc.values                           # float (nlat, nlon)

    # Convert GFS 0-360° longitude to -180–180° and sort
    lons_180 = np.where(lons_g > 180, lons_g - 360, lons_g)
    sort_idx = np.argsort(lons_180)
    lons_180 = lons_180[sort_idx]
    data_g   = data_g[:, sort_idx]

    # ── Trajectory bounds & map center ────────────────────────────────────────
    traj_lats = [s["lat"] for s in trajectory]
    traj_lons = [s["lon"] for s in trajectory]
    lat_c = (min(traj_lats) + max(traj_lats)) / 2
    lon_c = (min(traj_lons) + max(traj_lons)) / 2

    lat_span = max(traj_lats) - min(traj_lats)
    lon_span = max(traj_lons) - min(traj_lons)
    pad = max(max(lat_span, lon_span) * 0.6, 1.5)   # at least ±1.5°

    lon_min, lon_max = lon_c - pad, lon_c + pad
    lat_min, lat_max = lat_c - pad, lat_c + pad

    # ── Subset cloud cover to the plot window (faster rendering) ──────────────
    lat_mask = (lats_g >= lat_min - 0.5) & (lats_g <= lat_max + 0.5)
    lon_mask = (lons_180 >= lon_min - 0.5) & (lons_180 <= lon_max + 0.5)
    lats_sub = lats_g[lat_mask]
    lons_sub = lons_180[lon_mask]
    data_sub = data_g[np.ix_(lat_mask, lon_mask)]

    # ── Geostationary projection centered on trajectory midpoint ──────────────
    proj = ccrs.Geostationary(central_longitude=lon_c)
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": proj})
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # ── Cloud cover heatmap ───────────────────────────────────────────────────
    lon2d, lat2d = np.meshgrid(lons_sub, lats_sub)
    pcm = ax.pcolormesh(
        lon2d, lat2d, data_sub,
        cmap="Blues", vmin=0, vmax=100,
        transform=ccrs.PlateCarree(), shading="auto", zorder=1,
    )
    cbar = plt.colorbar(pcm, ax=ax, orientation="vertical", fraction=0.03, pad=0.04)
    cbar.set_label("Total Cloud Cover (%)", fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    # ── Map features ──────────────────────────────────────────────────────────
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="black", zorder=3)
    ax.add_feature(cfeature.BORDERS,   linewidth=1.2, edgecolor="black", zorder=3)
    ax.add_feature(cfeature.STATES,    linewidth=0.5, edgecolor="dimgray", zorder=3, linestyle=":")
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True,
        linewidth=0.5, color="gray", alpha=0.6, linestyle="--", zorder=2,
    )
    gl.top_labels    = False
    gl.right_labels  = False
    gl.xlabel_style  = {"size": 14}
    gl.ylabel_style  = {"size": 14}

    # ── Trajectory line & waypoint dots ──────────────────────────────────────
    pc = ccrs.PlateCarree()
    ax.plot(traj_lons, traj_lats, "-", color="tomato",
            linewidth=2.5, transform=pc, zorder=5)
    ax.plot(traj_lons, traj_lats, "o", color="white",
            markersize=7, markeredgecolor="tomato", markeredgewidth=1.5,
            transform=pc, zorder=6)

    # Start (green triangle) and end (red square)
    ax.plot(traj_lons[0], traj_lats[0], "^", color="limegreen",
            markersize=13, markeredgecolor="darkgreen", markeredgewidth=1,
            transform=pc, zorder=7, label="Start")
    ax.plot(traj_lons[-1], traj_lats[-1], "s", color="tomato",
            markersize=11, markeredgecolor="darkred", markeredgewidth=1,
            transform=pc, zorder=7, label="End")

    # Time labels at each waypoint
    for s in trajectory:
        h = s["time_min"] // 60
        m = s["time_min"] % 60
        ax.text(
            s["lon"] + pad * 0.04, s["lat"] + pad * 0.04,
            f"{h:02d}:{m:02d}",
            transform=pc, fontsize=14, color="darkred",
            fontweight="bold", zorder=8,
        )

    # ── Title ─────────────────────────────────────────────────────────────────
    total_min = trajectory[-1]["time_min"]
    h_tot, m_tot = total_min // 60, total_min % 60
    ax.set_title(
        f"Balloon Trajectory  —  {wx['altitude_m']:.0f} m ASL / {wx['pressure_hpa']:.1f} hPa\n"
        f"GFS Total Cloud Cover  |  Duration {h_tot:02d}:{m_tot:02d}  |  "
        f"Drift {trajectory[-1]['dist_km']:.1f} km",
        fontsize=14,
    )
    ax.legend(loc="lower left", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Map saved → {output_path}")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch NOMADS/GFS weather at a lat/lon/altitude and simulate\n"
            "balloon drift in N × 30-minute steps (constant altitude)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python balloon_weather.py 39.7392 -104.9903 3000\n"
            "  python balloon_weather.py 39.7392 -104.9903 3000 6\n"
            "  python balloon_weather.py 35.0 -97.0 10000 12\n"
        ),
    )
    parser.add_argument("lat",      type=float, help="Latitude  (°N; negative = S)")
    parser.add_argument("lon",      type=float, help="Longitude (°E; negative = W)")
    parser.add_argument("altitude", type=float, help="Altitude  (m above sea level)")
    parser.add_argument(
        "n_steps", type=int, nargs="?", default=6,
        help="Number of 30-min steps to simulate (default: 1)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip the map; print trajectory table only"
    )
    parser.add_argument(
        "--output", default="balloon_trajectory.png", metavar="FILE",
        help="Output image path (default: balloon_trajectory.png)"
    )
    args = parser.parse_args()

    H, wx = fetch_weather(args.lat, args.lon, args.altitude)
    print_weather(args.lat, args.lon, wx)

    trajectory = compute_trajectory(H, args.lat, args.lon, wx, args.n_steps)
    print_trajectory(trajectory, wx["altitude_m"], wx["pressure_hpa"])

    if not args.no_plot:
        fig = plot_trajectory(H, trajectory, wx, output_path=args.output)
        plt.show()


if __name__ == "__main__":
    main()
