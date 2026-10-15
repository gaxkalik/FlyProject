"""
Real-Time Detailed Weather Ingestion and Micro-Climate Downscaling.
Connects to live high-resolution meteorological telemetry (Open-Meteo API)
and applies atmospheric boundary-layer downscaling (Prandtl-von Kármán law)
to generate biologically accurate physical fields for Drosophila flight.
"""

import os
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import numpy as np

@dataclass
class RealTimeWeatherReading:
    timestamp_iso: str
    latitude: float
    longitude: float
    elevation_m: float
    temperature_c: float          # Ambient temperature (deg C)
    relative_humidity_pct: float  # Relative humidity (%)
    surface_pressure_kpa: float   # Barometric pressure (kPa)
    wind_speed_ms: float          # Wind speed at 10m (m/s)
    wind_direction_deg: float     # Direction wind is coming from (degrees)
    wind_gust_ms: float           # Wind gusts (m/s)
    d_temp_dt: float = 0.0        # Rate of temperature change (deg C / s)
    d_humidity_dt: float = 0.0    # Rate of humidity change (% / s)
    d_pressure_dt: float = 0.0    # Rate of pressure change (kPa / s)
    wind_vector_10m: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

class RealTimeWeatherClient:
    """
    Fetches, downscales, and streams real-time high-resolution weather data.
    """

    def __init__(
        self,
        latitude: float = 40.18,      # Default: Yerevan (UTC+4 local time)
        longitude: float = 44.51,
        cache_file: str = "data/live_weather_cache.json"
    ):
        self.lat = latitude
        self.lon = longitude
        self.cache_file = os.path.abspath(cache_file)
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

        # Atmospheric boundary layer roughness length (meters)
        # z0 ~ 0.005 m for open field/meadow where fruit flies navigate
        self.z0 = 0.005

    def fetch_live_weather(self, force_live: bool = False) -> RealTimeWeatherReading:
        """
        Fetches current conditions from Open-Meteo API or falls back to local cache.
        """
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={self.lat}&longitude={self.lon}"
            f"&current=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m"
            f"&minutely_15=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_gusts_10m"
        )

        data = None
        # Attempt live network request
        if force_live or not os.path.exists(self.cache_file):
            try:
                import ssl
                import certifi
                ctx = ssl.create_default_context(cafile=certifi.where())
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "FlyMesh-Connectome-Simulation/1.0"}
                )
                with urllib.request.urlopen(req, context=ctx, timeout=5) as response:
                    raw_json = response.read().decode("utf-8")
                    data = json.loads(raw_json)
                    with open(self.cache_file, "w") as f:
                        f.write(raw_json)
            except Exception as e:
                # Fallback to cache if offline
                if os.path.exists(self.cache_file):
                    with open(self.cache_file, "r") as f:
                        data = json.load(f)
                else:
                    raise RuntimeError(f"Failed to fetch live weather and no cache found: {e}")
        else:
            with open(self.cache_file, "r") as f:
                data = json.load(f)

        # Parse telemetry
        current = data["current"]
        temp_c = float(current["temperature_2m"])
        hum_pct = float(current["relative_humidity_2m"])
        # Open-Meteo surface_pressure is in hPa -> convert to kPa
        p_kpa = float(current["surface_pressure"]) / 10.0
        # Wind speed is in km/h -> convert to m/s
        wind_ms = float(current["wind_speed_10m"]) / 3.6
        wind_dir_deg = float(current["wind_direction_10m"])
        gust_ms = float(current["wind_gusts_10m"]) / 3.6

        # Convert meteorological wind direction (direction FROM) to cartesian velocity vector (direction TO)
        wind_rad = np.radians(wind_dir_deg)
        # In meteorology: 0 deg = North (wind blowing South), 90 deg = East (blowing West)
        # Cartesian vector: vx = -wind_ms * sin(theta), vy = -wind_ms * cos(theta)
        vx = -wind_ms * np.sin(wind_rad)
        vy = -wind_ms * np.cos(wind_rad)
        wind_vector = np.array([vx, vy, 0.0], dtype=np.float32)

        # Dynamic derivatives from 15-minute series
        minutely = data.get("minutely_15", {})
        times = minutely.get("time", [])
        d_temp_dt = 0.0
        d_hum_dt = 0.0
        d_press_dt = 0.0

        if len(times) >= 2:
            temps = minutely["temperature_2m"]
            hums = minutely["relative_humidity_2m"]
            press = [p / 10.0 for p in minutely["surface_pressure"]]
            # Derivative over 15 minutes (900 seconds)
            dt_step = 900.0
            idx = min(len(temps) - 1, 72)  # approximate current slot
            d_temp_dt = (temps[idx] - temps[idx - 1]) / dt_step
            d_hum_dt = (hums[idx] - hums[idx - 1]) / dt_step
            d_press_dt = (press[idx] - press[idx - 1]) / dt_step

        return RealTimeWeatherReading(
            timestamp_iso=current["time"],
            latitude=float(data["latitude"]),
            longitude=float(data["longitude"]),
            elevation_m=float(data["elevation"]),
            temperature_c=temp_c,
            relative_humidity_pct=hum_pct,
            surface_pressure_kpa=p_kpa,
            wind_speed_ms=wind_ms,
            wind_direction_deg=wind_dir_deg,
            wind_gust_ms=gust_ms,
            d_temp_dt=d_temp_dt,
            d_humidity_dt=d_hum_dt,
            d_pressure_dt=d_press_dt,
            wind_vector_10m=wind_vector
        )

    def downscale_to_insect_boundary_layer(
        self,
        reading: RealTimeWeatherReading,
        z_heights_m: np.ndarray
    ) -> np.ndarray:
        """
        Applies the logarithmic wind profile (Prandtl-von Kármán boundary layer law)
        to compute realistic micro-scale wind speeds at fly flight altitudes (0.1m - 2.0m).
        u(z) = u_10 * ln(z / z0) / ln(10.0 / z0)
        """
        z_clipped = np.maximum(self.z0 * 1.5, z_heights_m)
        log_profile = np.log(z_clipped / self.z0) / np.log(10.0 / self.z0)
        log_profile = np.clip(log_profile, 0.05, 1.0)

        # Scale 10m wind vector to height z
        # shape: [N, 3]
        scaled_wind = reading.wind_vector_10m[np.newaxis, :] * log_profile[:, np.newaxis]
        return scaled_wind.astype(np.float32)
