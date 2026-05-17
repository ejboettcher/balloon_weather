# balloon_weather

Fetch real-time NOAA GFS weather data for a lat/lon/altitude and predict where
a balloon will be 30 minutes later based on model winds.

Data source: **NOMADS GFS 0.25°** via OPeNDAP — no API key required.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python balloon_weather.py <lat> <lon> <altitude_m>
```

| Argument | Description |
|----------|-------------|
| `lat` | Latitude in degrees N (negative = S) |
| `lon` | Longitude in degrees E (negative = W) |
| `altitude_m` | Altitude in meters above sea level |

### Examples

```bash
# Denver, CO at 3 km
python balloon_weather.py 39.7392 -104.9903 3000

# Oklahoma City at 10 km (near tropopause)
python balloon_weather.py 35.0 -97.0 10000
```

### Sample output

```
GFS run : 2026-05-17 06:00 UTC
Source  : https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs20260517/gfs_0p25_06z
Grid pt : 39.750°N, 255.000°E
Level   : 3000 m → 700.0 hPa (model 2983 m)

──────────────────────────────────────────────────────
  BALLOON WEATHER REPORT  (NOMADS GFS 0.25°)
──────────────────────────────────────────────────────
  Query : +39.7392°  -104.9903°  3000 m ASL
  Model : +39.7500°  +255.0000°  2983 m  (700.0 hPa)
──────────────────────────────────────────────────────
  Temperature   : -4.3 °C
  Rel. Humidity : 52.1 %
  Cloud Cover   : 23.5 %  (total column)
  Wind (U, E+)  : +8.42 m/s
  Wind (V, N+)  : +2.11 m/s
  Wind speed    : 8.68 m/s  (16.9 kt)
  Wind from     : 256°
──────────────────────────────────────────────────────
  Predicted position after 30 min (wind drift, constant altitude):
    Latitude  : +39.76205°
    Longitude : -104.77902°
    Drift     : 9.49 km  (5.1 nm)
──────────────────────────────────────────────────────
```

## How it works

1. Discovers the latest available GFS 0.25° run on NOMADS (typically ~4 h lag).
2. Opens the OPeNDAP endpoint with `xarray` + `pydap` — streams only the needed slices.
3. Maps the requested altitude to the nearest model pressure level using the **geopotential height** field.
4. Extracts temperature, RH, cloud cover, and U/V wind components.
5. Dead-reckons the balloon position for 30 minutes:

```
Δlat = V × Δt / R
Δlon = U × Δt / (R × cos(lat))
```

Cloud cover is the **total-column** value reported by GFS — it represents the
sky fraction integrated over the entire atmospheric column, not just the queried level.
