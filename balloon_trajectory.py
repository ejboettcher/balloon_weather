#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Predict where a balloon will drift over the next several hours, given a
starting latitude, longitude, and altitude, using GFS wind forecasts
pulled via Herbie.

Method
------
This is a simple Lagrangian parcel trajectory (a "poor man's balloon
trajectory model"):

  1. Convert the balloon's altitude to a pressure level (hPa) using the
     standard atmosphere approximation.
  2. For each forecast hour, download the U/V wind at that pressure level
     from GFS (via Herbie), interpolate it to the balloon's current
     lat/lon, and step the balloon's position forward by one hour using
     that wind vector (a simple Euler advection step).
  3. Repeat, re-deriving the pressure level each step if you provide an
     ascent/descent rate, otherwise holding altitude constant.

Caveats (read before trusting this for anything real):
  - This ignores vertical motion's effect on wind shear unless you set
    a nonzero ascent/descent rate.
  - Real balloon trajectories are also very sensitive to super-pressure
    balloon float dynamics, small-scale turbulence, and model error --
    GFS is a global ~25 km model and will smooth out local wind features
    that matter close to the ground or near terrain.
  - This uses simple Euler stepping (1-hour steps). For rigorous work,
    use smaller sub-steps or RK4 and validate against known flights.
  - Do not use this for real flight safety decisions (e.g. actual
    manned balloon or aviation operations) without a validated,
    professional trajectory model.
"""

import argparse
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from herbie import Herbie

import matplotlib
matplotlib.use("Agg")  # headless-safe backend, works even with no display
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

EARTH_RADIUS_M = 6_371_000.0


def pressure_from_altitude_m(alt_m: float) -> float:
    """
    Approximate pressure (hPa) from geometric altitude (m) using the
    ICAO standard atmosphere (valid roughly 0-11 km, i.e. within the
    troposphere -- fine for weather balloons in ascent, not for the
    stratosphere).
    """
    return 1013.25 * (1 - 2.25577e-5 * alt_m) ** 5.25588


def nearest_gfs_pressure_level(pressure_hpa: float) -> int:
    """Snap to the nearest GFS isobaric level available in pgrb2.0p25."""
    levels = [
        1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700,
        675, 650, 625, 600, 575, 550, 525, 500, 475, 450, 425, 400, 350,
        300, 250, 200, 150, 100, 70, 50, 30, 20, 10,
    ]
    return min(levels, key=lambda lv: abs(lv - pressure_hpa))


def latest_available_gfs_run(now: datetime | None = None) -> datetime:
    """
    GFS runs at 00/06/12/18Z and takes ~3.5-4.5 hours to become fully
    available on NOMADS/AWS. Pick the most recent run that should
    already be posted.

    Returns a naive datetime (implicitly UTC) since Herbie and pandas
    internals expect naive timestamps -- mixing tz-aware and naive
    datetimes raises "can't compare offset-naive and offset-aware
    datetimes".
    """
    if now is None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
    now = now - timedelta(hours=4, minutes=30)  # availability lag buffer
    run_hour = (now.hour // 6) * 6
    return now.replace(hour=run_hour, minute=0, second=0, microsecond=0)


def get_uv_at_point(run_time: datetime, fxx: int, level_hpa: int,
                     lat: float, lon_0_360: float) -> tuple[float, float]:
    """
    Fetch U/V wind (m/s) at a given pressure level and forecast hour,
    interpolated to (lat, lon). Returns (u_east, v_north) in m/s.
    """
    H = Herbie(
        run_time,
        model="gfs",
        product="pgrb2.0p25",
        fxx=fxx,
        verbose=False,
    )

    search = f":[UV]GRD:{level_hpa} mb:"
    ds = H.xarray(search, remove_grib=True)

    if isinstance(ds, list):
        if len(ds) == 0:
            raise RuntimeError(
                f"No UGRD/VGRD data returned for fxx={fxx}, level={level_hpa} mb."
            )
        ds = ds[0]

    points = pd.DataFrame({"longitude": [lon_0_360], "latitude": [lat]})
    picked = ds.herbie.pick_points(points, method="nearest")

    u = float(picked["u"].values.squeeze())
    v = float(picked["v"].values.squeeze())
    return u, v


def step_position(lat: float, lon: float, u_ms: float, v_ms: float,
                   dt_s: float) -> tuple[float, float]:
    """
    Advance a lat/lon position forward by dt_s seconds given a constant
    wind vector (u = eastward m/s, v = northward m/s). Uses a simple
    flat-Earth approximation valid for short (~hour-scale) time steps.
    """
    dy_m = v_ms * dt_s
    dx_m = u_ms * dt_s

    dlat = (dy_m / EARTH_RADIUS_M) * (180.0 / math.pi)
    dlon = (dx_m / (EARTH_RADIUS_M * math.cos(math.radians(lat)))) * (180.0 / math.pi)

    return lat + dlat, lon + dlon


def predict_trajectory(lat0: float, lon0: float, alt0_m: float,
                        hours: int = 6, step_hours: int = 1,
                        ascent_rate_m_per_hr: float = 0.0,
                        run_time: datetime | None = None) -> pd.DataFrame:
    """
    Predict the balloon's trajectory over `hours` hours, stepping every
    `step_hours`. Returns a DataFrame with one row per time step.
    """
    if run_time is None:
        run_time = latest_available_gfs_run()

    lat, lon = lat0, lon0 % 360.0  # GFS grid is 0-360 longitude
    alt_m = alt0_m

    rows = [{
        "fxx": 0,
        "valid_time": run_time,
        "lat": lat,
        "lon": ((lon + 180) % 360) - 180,  # back to -180..180 for display
        "alt_m": alt_m,
        "pressure_hpa": pressure_from_altitude_m(alt_m),
        "u_ms": np.nan,
        "v_ms": np.nan,
    }]

    n_steps = hours // step_hours
    dt_s = step_hours * 3600

    for step in range(1, n_steps + 1):
        fxx = step * step_hours
        pressure = pressure_from_altitude_m(alt_m)
        level = nearest_gfs_pressure_level(pressure)

        u, v = get_uv_at_point(run_time, fxx, level, lat, lon)
        lat, lon = step_position(lat, lon, u, v, dt_s)
        alt_m += ascent_rate_m_per_hr * step_hours

        rows.append({
            "fxx": fxx,
            "valid_time": run_time + timedelta(hours=fxx),
            "lat": lat,
            "lon": ((lon + 180) % 360) - 180,
            "alt_m": alt_m,
            "pressure_hpa": pressure,
            "u_ms": u,
            "v_ms": v,
        })

    return pd.DataFrame(rows)


def get_cloud_cover(run_time: datetime, fxx: int):
    """
    Fetch total cloud cover (%) for the whole atmosphere column from GFS
    at a given forecast hour, returned as an xarray DataArray with
    longitude in -180..180 and sorted ascending on both lat and lon.
    """
    H = Herbie(
        run_time,
        model="gfs",
        product="pgrb2.0p25",
        fxx=fxx,
        verbose=False,
    )
    ds = H.xarray(":TCDC:entire atmosphere:", remove_grib=True)

    # If the search string matches more than one GRIB message (e.g. GFS
    # sometimes has both an instantaneous and a time-averaged TCDC record
    # for the same level), Herbie can't merge them into one Dataset and
    # returns a list instead. Just take the first one in that case.
    if isinstance(ds, list):
        if len(ds) == 0:
            raise RuntimeError(
                f"No TCDC data returned for fxx={fxx}. "
                "Try checking H.inventory() to see what's actually available."
            )
        ds = ds[0]

    ds = ds.herbie.to_180()
    ds = ds.sortby("latitude").sortby("longitude")

    # Cloud cover is the only field returned by this search; grab whatever
    # cfgrib decided to name it rather than hardcoding (varies by version).
    varname = list(ds.data_vars)[0]
    return ds[varname]


def plot_map_panel(ax, cloud_da, traj_df, upto_idx: int, title: str, pad_deg: float = 4.0):
    """
    Draw one map panel: cloud cover shading, coastlines/borders, the full
    trajectory as a line, the starting point, and the balloon's position
    as of `upto_idx` in traj_df.
    """
    lat_min, lat_max = traj_df["lat"].min() - pad_deg, traj_df["lat"].max() + pad_deg
    lon_min, lon_max = traj_df["lon"].min() - pad_deg, traj_df["lon"].max() + pad_deg
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    mesh = ax.pcolormesh(
        cloud_da["longitude"], cloud_da["latitude"], cloud_da.values,
        transform=ccrs.PlateCarree(), cmap="Blues", vmin=0, vmax=100, shading="auto",
    )

    ax.coastlines(resolution="50m", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    ax.add_feature(cfeature.STATES, linewidth=0.3)

    ax.plot(traj_df["lon"], traj_df["lat"], color="red", linewidth=2,
             transform=ccrs.PlateCarree(), zorder=5, label="Predicted path")
    ax.plot(traj_df["lon"].iloc[0], traj_df["lat"].iloc[0], marker="o", color="lime",
             markersize=10, markeredgecolor="black", transform=ccrs.PlateCarree(),
             zorder=6, label="Start")
    ax.plot(traj_df["lon"].iloc[upto_idx], traj_df["lat"].iloc[upto_idx], marker="*",
             color="yellow", markersize=18, markeredgecolor="black",
             transform=ccrs.PlateCarree(), zorder=6, label="Position shown")

    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    return mesh


def make_trajectory_maps(df: pd.DataFrame, run_time: datetime,
                          out_path: str = "balloon_trajectory_maps.png"):
    """
    Build a two-panel figure: cloud cover + trajectory at t=0 (left) and
    at the final forecast hour (right).
    """
    fxx_start = int(df["fxx"].iloc[0])
    fxx_final = int(df["fxx"].iloc[-1])

    print(f"Fetching cloud cover for f{fxx_start:03d} and f{fxx_final:03d}...")
    cloud_start = get_cloud_cover(run_time, fxx_start)
    cloud_final = get_cloud_cover(run_time, fxx_final)

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6.5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    t0_valid = df["valid_time"].iloc[0]
    tf_valid = df["valid_time"].iloc[-1]

    plot_map_panel(
        axes[0], cloud_start, df, upto_idx=0,
        title=f"t=0  ({t0_valid:%Y-%m-%d %H:%M} UTC)",
    )
    mesh = plot_map_panel(
        axes[1], cloud_final, df, upto_idx=len(df) - 1,
        title=f"t=+{fxx_final}h  ({tf_valid:%Y-%m-%d %H:%M} UTC)",
    )

    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal",
                         fraction=0.05, pad=0.08, shrink=0.6)
    cbar.set_label("Total cloud cover (%)")

    fig.suptitle("Predicted Balloon Trajectory Over GFS Cloud Cover", fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trajectory maps to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=float, required=True, help="Starting latitude (deg)")
    parser.add_argument("--lon", type=float, required=True, help="Starting longitude (deg, -180 to 180)")
    parser.add_argument("--alt", type=float, required=True, help="Starting altitude (meters above sea level)")
    parser.add_argument("--hours", type=int, default=6, help="Hours to predict forward (default: 6)")
    parser.add_argument("--step-hours", type=int, default=1, help="Step size in hours (default: 1)")
    parser.add_argument("--ascent-rate", type=float, default=0.0,
                         help="Ascent rate in m/hr (negative for descent; default: 0, constant altitude)")
    parser.add_argument("--no-plot", action="store_true",
                         help="Skip generating the cloud-cover trajectory maps")
    args = parser.parse_args()

    print(f"Starting position: lat={args.lat}, lon={args.lon}, alt={args.alt} m")
    print(f"Using GFS run: {latest_available_gfs_run()} UTC\n")

    df = predict_trajectory(
        lat0=args.lat,
        lon0=args.lon,
        alt0_m=args.alt,
        hours=args.hours,
        step_hours=args.step_hours,
        ascent_rate_m_per_hr=args.ascent_rate,
    )

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(df.to_string(index=False))

    final = df.iloc[-1]
    print(f"\nPredicted position after {args.hours} hours:")
    print(f"  lat={final['lat']:.4f}, lon={final['lon']:.4f}, alt={final['alt_m']:.0f} m")

    out_path = "balloon_trajectory.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved full trajectory to {out_path}")

    if not args.no_plot:
        make_trajectory_maps(df, latest_available_gfs_run())


if __name__ == "__main__":
    main()